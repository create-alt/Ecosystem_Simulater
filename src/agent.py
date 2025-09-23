import torch
import torch.optim as optim
from torch.distributions import Categorical
from models import ActorCritic

class Agent:
    def __init__(self, input_channels, action_num, config):
        """
        強化学習エージェント

        Args:
            input_channels (int): モデルに入力するチャネル数
            action_num (int): 行動の数
            config: 設定オブジェクト
        """
        self.config = config
        self.device = config.DEVICE
        
        self.model = ActorCritic(input_channels, action_num).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        
        self.saved_log_probs = []
        self.rewards = []
        self.state_values = []

    def select_actions(self, states, masks):
        """
        複数の個体の状態(観測)と行動マスクから、行動をバッチで選択する。

        Args:
            states (torch.Tensor): 観測情報のバッチ
            masks (torch.Tensor): 行動マスクのバッチ

        Returns:
            torch.Tensor: 選択された行動のインデックスのバッチ
        """
        if states.dim() == 3: # バッチサイズが1の場合
            states = states.unsqueeze(0)
        
        # policyの合計が0になるケース(移動不可かつ他の選択肢がないなど)の対策
        policy, state_value = self.model(states)
        policy = policy * masks
        # マスキング後、合計が0になった場合は、マスクを無視して再度計算
        if policy.sum(dim=1).min() == 0:
            bad_indices = (policy.sum(dim=1) == 0)
            policy_raw, _ = self.model(states[bad_indices])
            policy[bad_indices] = policy_raw

        # 行動を確率的に選択
        m = Categorical(policy)
        actions = m.sample()
        
        # ログ保存
        self.saved_log_probs.append(m.log_prob(actions))
        self.state_values.append(state_value)
        
        return actions

    def optimize_model(self):
        """
        収集した報酬とログからモデルを更新する (A2Cアルゴリズム)
        """
        if not self.saved_log_probs:
            return

        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        # 割引報酬を計算
        for r in self.rewards[::-1]:
            R = r + self.config.GAMMA * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device)
        
        # ステップ数が1の場合、stdが0になりnanが発生するのを防ぐ
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # 各ステップの損失を計算
        for log_prob, value, R in zip(self.saved_log_probs, self.state_values, returns):
            # Rはスカラー、valueは(num_organisms, 1)のテンソル
            # .item()を削除し、ブロードキャストを利用してAdvantageを計算
            advantage = R - value 
            
            # Actor(Policy)の損失
            # advantageは(num_organisms, 1)なので、log_prob (num_organisms,)と形状を合わせる
            # また、advantageの勾配がactorに伝播しないようにdetach()する
            # 個体ごとの損失を計算し、その平均を取る
            policy_losses.append((-log_prob * advantage.squeeze().detach()).mean())
            
            # Critic(Value)の損失
            # valueと同じ形状のターゲットテンソルを作成
            target_values = torch.full_like(value, R.item())
            value_losses.append(torch.nn.functional.smooth_l1_loss(value, target_values))

        self.optimizer.zero_grad()
        
        # 損失を合算して勾配を計算
        # 更新期間全体の平均損失を計算
        loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
        loss.backward()
        
        self.optimizer.step()
        
        # 次のイテレーションのためにデータをクリア
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.state_values[:]

    def save_model(self, path):
        """モデルの重みを保存"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """モデルの重みを読み込み"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))