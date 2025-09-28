import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_num):
        """
        Actor-Criticモデル

        Args:
            input_channels (int): 入力チャネル数（観測情報の種類数）
            action_num (int): 出力する行動の数
        """
        super(ActorCritic, self).__init__()
        # 観測範囲のサイズ ((2n+1)x(2n+1)の正方形)
        obs_size = 2 * Config.OBSERVATION_RANGE + 1

        # CNNを用いて観測範囲の特徴を抽出
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # CNNからの出力サイズを計算
        flattened_size = 32 * obs_size * obs_size

        # 全結合層
        self.fc1 = nn.Linear(flattened_size, 128)

        # Actor: 行動の確率分布を出力
        self.actor_head = nn.Linear(128, action_num)
        
        # Critic: 状態の価値を出力
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        順伝播

        Args:
            x (torch.Tensor): 観測情報のバッチ (batch_size, channels, height, width)

        Returns:
            torch.Tensor: 各行動の確率分布 (action_policy)
            torch.Tensor: 状態の価値 (state_value)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        # Actor: Softmaxで行動確率を計算
        action_policy = F.softmax(self.actor_head(x), dim=-1)
        
        # Critic: 状態価値を計算
        state_value = self.critic_head(x)

        return action_policy, state_value