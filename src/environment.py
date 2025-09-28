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
        self.max_plant = config.ENV_SIZE * config.ENV_SIZE / 5
        self.num_plant = 0
        self.grid = torch.zeros((4, self.size, self.size), dtype=torch.float32, device=self.device)
        self.herbivores = []
        self.carnivores = []

    # --- NEW RESET METHOD AND HELPER ---
    def reset(self):
        """
        環境を初期化。
        草食動物と肉食動物を、それぞれ所定のグループ数で群れを形成するように配置する。
        """
        self.grid.zero_()
        self.herbivores.clear()
        self.carnivores.clear()
        
        # 草食動物を5グループで生成
        self.herbivores = self._create_grouped_organisms(
            self.config.INIT_HERBIVORES, 10, Herbivore, 2
        )

        # 肉食動物を3グループで生成
        self.carnivores = self._create_grouped_organisms(
            self.config.INIT_CARNIVORES, 10, Carnivore, 3
        )

        self._spawn_plants()

    def _create_grouped_organisms(self, num_to_spawn, num_groups, SpeciesClass, grid_id):
        """
        指定された個体数の生物を、グループに分けて配置する。
        1. グループの中心点をランダムに決める。
        2. 個体数を各グループに割り振る。
        3. 各グループの中心点の周辺に個体を配置する。
        """
        organisms = []
        if num_to_spawn == 0:
            return organisms

        # 1. グループの中心点を決定
        group_centers = []
        for _ in range(num_groups):
            center = self._get_random_empty_pos()
            if center[0] is not None:
                group_centers.append(center)
        
        if not group_centers: # 中心点を置くスペースすらない場合
            return organisms

        # 2. 個体数を各グループに割り振る
        inds_per_group = [num_to_spawn // len(group_centers)] * len(group_centers)
        for i in range(num_to_spawn % len(group_centers)):
            inds_per_group[i] += 1
        
        # 3. 各グループの中心点周辺に個体を配置
        for center, count in zip(group_centers, inds_per_group):
            for _ in range(count):
                x, y = self._get_random_empty_pos_nearby(
                    center[0], center[1], search_radius=self.config.GROUP_SPAWN_RADIUS
                )
                if x is not None:
                    org = SpeciesClass(x, y, self.config)
                    organisms.append(org)
                    self.grid[grid_id, y, x] = 1 # 配置したマスを埋める
                else:
                    # 近くに空きがない場合、配置をスキップ（環境が満杯に近いケース）
                    continue
        
        return organisms

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

        num_to_spawn = random.randint(1, max(1, len(empty_cells) // 2))
        indices = random.sample(range(len(empty_cells)), k=num_to_spawn)
        
        for idx in indices:
            if self.num_plant >= self.max_plant:
                break
            y, x = empty_cells[idx]
            self.grid[1, y, x] = 1
            self.num_plant+=1

    def get_batch_observations(self, organisms):
        observations = [get_observation_tensor(self.grid, org.x, org.y, self.config.OBSERVATION_RANGE, self.device) for org in organisms]
        return torch.stack(observations)

    def get_batch_masks(self, organisms):
        masks = [get_action_mask(self.grid, org.x, org.y, self.device) for org in organisms]
        return torch.stack(masks)

    def step(self, herbivore_actions, carnivore_actions):
        """1シミュレーションステップを実行"""
        rewards = { "herbivore": {}, "carnivore": {} }
        self._execute_actions(self.herbivores, herbivore_actions, rewards["herbivore"])
        self._execute_actions(self.carnivores, carnivore_actions, rewards["carnivore"])
        self._handle_reproduction()
        dead_organisms = self._handle_deaths()
        self._apply_kin_penalties(dead_organisms, rewards)
        self.herbivores = [org for org in self.herbivores if org.is_alive]
        self.carnivores = [org for org in self.carnivores if org.is_alive]
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
                self.num_plant-=1
                self.grid[1, new_y, new_x] = 0
                reward += self.config.REWARD_EAT
                can_move = True
            elif isinstance(org, Carnivore) and target_obj_id == 2:
                target_org = self._find_organism_at(new_x, new_y, include_dead=True)
                if target_org and target_org.is_alive:
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
                
    def _find_organism_at(self, x, y, include_dead=False):
        """指定座標の生物インスタンスを探す"""
        for org in self.herbivores + self.carnivores:
            if org.x == x and org.y == y:
                if include_dead or org.is_alive:
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
            if dead_org.death_cause != 'old_age' and dead_org.parent_ids is not None:
                all_living_orgs = self.herbivores + self.carnivores
                for kin in all_living_orgs:
                    if kin.is_alive and kin.parent_ids is not None and kin.id != dead_org.id:
                        if any(pid in kin.parent_ids for pid in dead_org.parent_ids):
                            if isinstance(kin, Herbivore):
                                species_rewards = rewards["herbivore"]
                            else:
                                species_rewards = rewards["carnivore"]
                            if kin.id in species_rewards:
                                species_rewards[kin.id] += self.config.PENALTY_KIN_DEATH
                            else:
                                species_rewards[kin.id] = self.config.PENALTY_KIN_DEATH
    
    def _handle_reproduction(self):
        """種族ごとに有性生殖を処理する"""
        self._reproduce_species(self.herbivores, Herbivore, 2)
        self._reproduce_species(self.carnivores, Carnivore, 3)

    def _reproduce_species(self, organisms, species_class, grid_id):
        """指定された種族のリストで有性生殖を行う"""
        reproduced_this_step = set()
        new_offspring = []
        for org in list(organisms):
            if org.id in reproduced_this_step or not org.is_alive or org.satiety_level < self.config.REPRODUCTION_SATIETY_BOADER:
                continue
            if random.random() < self.config.REPRODUCTION_PROB:
                partner = self._find_partner(org, reproduced_this_step)
                if partner:
                    spawn_x, spawn_y = self._get_random_empty_pos_nearby(org.x, org.y)
                    if spawn_x is not None:
                        offspring = species_class(spawn_x, spawn_y, self.config, parent_ids=(org.id, partner.id), init_satiety_level=int(0.5*(org.satiety_level + partner.satiety_level)))
                        new_offspring.append(offspring)
                        self.grid[grid_id, spawn_y, spawn_x] = 1
                        reproduced_this_step.add(org.id)
                        reproduced_this_step.add(partner.id)
        organisms.extend(new_offspring)

    def _find_partner(self, org, reproduced_ids):
        """指定個体の周囲8マスにいる繁殖可能なパートナーを探す"""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                search_x, search_y = org.x + dx, org.y + dy
                potential_partner = self._find_organism_at(search_x, search_y)
                if (potential_partner and 
                    isinstance(potential_partner, type(org)) and
                    potential_partner.id not in reproduced_ids):
                    return potential_partner
        return None

    def _get_random_empty_pos_nearby(self, x, y, search_radius=2):
        """指定座標の近くにある空きマスをランダムに返す"""
        empty_neighbors = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and self.grid[:, ny, nx].sum() == 0):
                    empty_neighbors.append((nx, ny))
        if empty_neighbors:
            return random.choice(empty_neighbors)
        else:
            return self._get_random_empty_pos()