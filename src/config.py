import torch

class Config:
    # デバイス設定 (GPUが利用可能ならGPUを使用)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 環境設定
    ENV_SIZE = 50  # 環境の1辺のサイズ (可視化のため少し小さく設定)

    # 生物の初期個体数
    INIT_HERBIVORES = 30
    INIT_CARNIVORES = 5

    # シミュレーション設定
    SIMULATION_STEPS = 1000

    # 生物のパラメータ
    MAX_AGE = 100
    MAX_SATIETY = 20
    MAX_FATIGUE = 20
    SATIETY_PER_ACTION = 1
    FATIGUE_PER_ACTION = 1
    SATIETY_FROM_EAT = 10
    FATIGUE_FROM_SLEEP = 10
    REPRODUCTION_PROB = 0.05 # 生殖の確率

    # 強化学習パラメータ
    LR = 0.001 # 学習率
    GAMMA = 0.99 # 割引率
    OBSERVATION_RANGE = 5 # 観測範囲（マンハッタン距離）

    # --- 報酬設定 (更新) ---
    REWARD_EAT = 5.0
    REWARD_SURVIVE = 0.01 # 基本生存報酬は小さく設定
    REWARD_REPRODUCE = 10.0
    REWARD_GROUP_SURVIVAL_RATE = 0.01 # グループ報酬係数（個体数ごとに加算）

    PENALTY_DEATH_HUNGER = -10.0
    PENALTY_DEATH_FATIGUE = -10.0
    PENALTY_KIN_DEATH = -5.0 # 親族の不自然死ペナルティ
    PENALTY_HUNGER_RATE = 0.05 # 空腹度ペナルティ係数
    PENALTY_FATIGUE_RATE = 0.05 # 疲労度ペナルティ係数