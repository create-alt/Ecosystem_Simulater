import torch

class Config:
    # デバイス設定 (GPUが利用可能ならGPUを使用)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 環境設定
    ENV_SIZE = 50  # 環境の1辺のサイズ

    # 生物の初期個体数
    INIT_HERBIVORES = 100
    INIT_CARNIVORES = 50

    # --- NEW: 学習・シミュレーション設定 ---
    # SIMULATION_STEPS は TRAINING_STEPS と VISUALIZATION_STEPS に分割
    TRAINING_STEPS = 2000          # 学習の総ステップ数
    VISUALIZATION_STEPS = 500      # 学習後に可視化するシミュレーションのステップ数

    # 生物のパラメータ
    MAX_AGE = 200
    MAX_SATIETY = 50
    MAX_FATIGUE = 50
    SATIETY_PER_ACTION = 1
    FATIGUE_PER_ACTION = 1
    SATIETY_FROM_EAT = 10
    REPRODUCTION_SATIETY_BOADER = MAX_SATIETY / 2
    FATIGUE_FROM_SLEEP = 10
    REPRODUCTION_PROB = 0.1 # 生殖の確率

    # --- NEW: 並列強化学習パラメータ ---
    NUM_ENVIRONMENTS = 10     # 並列実行する環境の数
    UPDATE_INTERVAL = 20      # Nステップごとにモデルを更新
    LR = 0.001                # 学習率
    GAMMA = 0.99              # 割引率
    OBSERVATION_RANGE = 10    # 観測範囲（マンハッタン距離）

    # --- 報酬設定 ---
    REWARD_EAT = 10.0
    REWARD_SURVIVE = 0.02
    REWARD_REPRODUCE = 10.0
    REWARD_GROUP_SURVIVAL_RATE = 0.01

    PENALTY_DEATH_HUNGER = -10.0
    PENALTY_DEATH_FATIGUE = -10.0
    PENALTY_KIN_DEATH = -5.0
    PENALTY_HUNGER_RATE = 0.05
    PENALTY_FATIGUE_RATE = 0.05

    # その他
    GROUP_SPAWN_RADIUS = 6