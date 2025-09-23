import torch
import random
from config import Config
from organism import Herbivore, Carnivore
from utility import get_observation_tensor, get_action_mask

class Env:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.size = config.ENV_SIZE
        self.grid = torch.zeros((4, self.size, self.size), dtype=torch.float32, device=self.device)
        self.herbivores = []
        self.carnivores = []

    def reset(self):
        """環境を初期化"""
        self.grid.zero_()
        self.herbivores.clear()
        self.carnivores.clear()

        # 草食動物を配置
        for _ in range(self.config.INIT_HERBIVORES):
            x, y = self._get_random_empty_pos()
            org = Herbivore(x, y, self.config)
            self.herbivores.append(org)
            self.grid[2, y, x] = 1

        # 肉食動物を配置
        for _ in range(self.config.INIT_CARNIVORES):
            x, y = self._get_random_empty_pos()
            org = Carnivore(x, y, self.config)
            self.carnivores.append(org)
            self.grid[3, y, x] = 1
            
        self._spawn_plants()

    def _get_random_empty_pos(self):
        """何もないランダムな座標を返す"""
        empty_cells = (self.grid.sum(dim=0) == 0).nonzero(as_tuple=False)
        if not len(empty_cells): return None, None
        y, x = random.choice(empty_cells)
        return x.item(), y.item()

    def _spawn_plants(self):
        """ランダムに植物を発生させる"""
        empty_cells = (self.grid.sum(dim=0) == 0).nonzero(as_tuple=False)
        if len(empty_cells) == 0: return

        num_to_spawn = random.randint(1, max(1, len(empty_cells) // 4))
        indices = random.sample(range(len(empty_cells)), k=num_to_spawn)
        
        for idx in indices:
            y, x = empty_cells[idx]
            self.grid[1, y, x] = 1

    def get_batch_observations(self, organisms):
        """指定された個体群の観測情報をバッチで取得"""
        observations = [get_observation_tensor(self.grid, org.x, org.y, self.config.OBSERVATION_RANGE, self.device) for org in organisms]
        return torch.stack(observations)

    def get_batch_masks(self, organisms):
        """指定された個体群の行動マスクをバッチで取得"""
        masks = [get_action_mask(self.grid, org.x, org.y, self.device) for org in organisms]
        return torch.stack(masks)

    def step(self, herbivore_actions, carnivore_actions):
        """1シミュレーションステップを実行"""
        rewards = { "herbivore": {}, "carnivore": {} }
        
        # 行動の実行と報酬計算
        self._execute_actions(self.herbivores, herbivore_actions, rewards["herbivore"])
        self._execute_actions(self.carnivores, carnivore_actions, rewards["carnivore"])
        
        # 生殖処理 (ペナルティ計算の前に実行)
        self._handle_reproduction()

        # 死亡処理と親族ペナルティ
        dead_organisms = self._handle_deaths()
        self._apply_kin_penalties(dead_organisms, rewards)

        # リストから死んだ個体を完全に削除
        self.herbivores = [org for org in self.herbivores if org.is_alive]
        self.carnivores = [org for org in self.carnivores if org.is_alive]
        
        # 植物の発生
        self._spawn_plants()
        
        return rewards

    def _execute_actions(self, organisms, actions, reward_dict):
        """個体群の行動を解決し、詳細な報酬を計算"""
        if not organisms: return

        num_group = len(organisms)

        for org, action_idx in zip(organisms, actions):
            if not org.is_alive: continue
            
            action = org.action_space[action_idx]
            
            reward = self.config.REWARD_SURVIVE
            reward += num_group * self.config.REWARD_GROUP_SURVIVAL_RATE
            reward -= (self.config.MAX_SATIETY - org.satiety_level) * self.config.PENALTY_HUNGER_RATE
            reward -= (self.config.MAX_FATIGUE - org.fatigue_level) * self.config.PENALTY_FATIGUE_RATE
            
            self.grid[org.type_id, org.y, org.x] = 0
            new_x, new_y = org.x, org.y
            
            if action == 0 and new_y > 0: new_y -= 1
            elif action == 1 and new_y < self.size - 1: new_y += 1
            elif action == 2 and new_x > 0: new_x -= 1
            elif action == 3 and new_x < self.size - 1: new_x += 1
            
            target_obj_id = self.grid[:, new_y, new_x].argmax().item() if self.grid[:, new_y, new_x].sum() > 0 else 0
            can_move = target_obj_id == 0
            
            if isinstance(org, Herbivore) and target_obj_id == 1:
                org.eat()
                self.grid[1, new_y, new_x] = 0
                reward += self.config.REWARD_EAT
                can_move = True
            elif isinstance(org, Carnivore) and target_obj_id == 2:
                target_org = self._find_organism_at(new_x, new_y)
                if target_org:
                    target_org.is_alive = False
                    target_org.death_cause = 'predation'
                    org.eat()
                    reward += self.config.REWARD_EAT
                    can_move = True

            if can_move:
                org.x, org.y = new_x, new_y
            
            org.update_status(action)
            reward_dict[org.id] = reward

            if org.is_alive:
                self.grid[org.type_id, org.y, org.x] = 1
                
    def _find_organism_at(self, x, y):
        """指定座標の生物インスタンスを探す"""
        # is_aliveのチェックを外して、死んだ直後の個体も見つけられるようにする
        for org in self.herbivores + self.carnivores:
            if org.x == x and org.y == y:
                return org
        return None

    def _handle_deaths(self):
        """死亡した個体のリストを返し、グリッドから削除する"""
        dead_this_step = []
        for org in self.herbivores + self.carnivores:
            if not org.is_alive and org.death_cause is not None:
                dead_this_step.append(org)
                if self.grid[org.type_id, org.y, org.x] == 1:
                    self.grid[org.type_id, org.y, org.x] = 0
        return dead_this_step

    def _apply_kin_penalties(self, dead_organisms, rewards):
        """親族に死亡ペナルティを適用"""
        for dead_org in dead_organisms:
            if dead_org.death_cause != 'old_age' and dead_org.parent_id is not None:
                all_living_orgs = self.herbivores + self.carnivores
                for kin in all_living_orgs:
                    if kin.is_alive and kin.parent_id == dead_org.parent_id and kin.id != dead_org.id:
                        
                        # --- ERROR FIX STARTS HERE ---
                        if isinstance(kin, Herbivore):
                            species_rewards = rewards["herbivore"]
                        else:
                            species_rewards = rewards["carnivore"]

                        # 報酬キーが存在すれば加算、なければ新規作成
                        if kin.id in species_rewards:
                            species_rewards[kin.id] += self.config.PENALTY_KIN_DEATH
                        else:
                            species_rewards[kin.id] = self.config.PENALTY_KIN_DEATH
                        # --- ERROR FIX ENDS HERE ---
    
    def _handle_reproduction(self):
        """生殖を処理する"""
        new_herbivores = []
        for org in self.herbivores:
            if org.is_alive and random.random() < self.config.REPRODUCTION_PROB:
                 x, y = self._get_random_empty_pos()
                 if x is not None:
                     new_org = Herbivore(x, y, self.config, parent_id=org.id)
                     new_herbivores.append(new_org)
                     self.grid[2, y, x] = 1
        self.herbivores.extend(new_herbivores)

        new_carnivores = []
        for org in self.carnivores:
            if org.is_alive and random.random() < self.config.REPRODUCTION_PROB:
                x, y = self._get_random_empty_pos()
                if x is not None:
                    new_org = Carnivore(x, y, self.config, parent_id=org.id)
                    new_carnivores.append(new_org)
                    self.grid[3, y, x] = 1
        self.carnivores.extend(new_carnivores)