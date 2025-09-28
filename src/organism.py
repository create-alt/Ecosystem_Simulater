import uuid

class Organism:
    """
    全ての生物の基底クラス。個体の状態を管理する。
    """
    def __init__(self, x, y, config, parent_ids=None, init_satiety_level=None): # parent_idからparent_idsに変更
        self.id = uuid.uuid4()
        self.parent_ids = parent_ids # 単一IDからIDのタプルに変更
        self.x = x
        self.y = y
        
        self.age = 0
        if init_satiety_level is None:
            self.satiety_level = config.MAX_SATIETY
        else:
            self.satiety_level = init_satiety_level
        self.fatigue_level = config.MAX_FATIGUE
        
        self.is_alive = True
        self.death_cause = None
        self.config = config

        self.action_space = list(range(6))
        self.action_num = len(self.action_space)

    def update_status(self, action):
        """1ステップ後の状態変化を計算"""
        self.age += 1
        
        if action == 4: # 睡眠
            self.satiety_level -= self.config.SATIETY_PER_ACTION
            self.fatigue_level = min(self.config.MAX_FATIGUE, self.fatigue_level + self.config.FATIGUE_FROM_SLEEP)
        else: # 睡眠以外の行動
            self.satiety_level -= self.config.SATIETY_PER_ACTION
            self.fatigue_level -= self.config.FATIGUE_PER_ACTION

        # 死亡判定
        if self.satiety_level <= 0:
            self.is_alive = False
            self.death_cause = 'starvation'
        elif self.fatigue_level <= 0:
            self.is_alive = False
            self.death_cause = 'fatigue'
        elif self.age > self.config.MAX_AGE:
            self.is_alive = False
            self.death_cause = 'old_age'

    def eat(self):
        """食事による状態変化"""
        self.satiety_level = min(self.config.MAX_SATIETY, self.satiety_level + self.config.SATIETY_FROM_EAT)

class Herbivore(Organism):
    """草食動物"""
    def __init__(self, x, y, config, parent_ids=None, init_satiety_level=None):
        super().__init__(x, y, config, parent_ids, init_satiety_level=init_satiety_level)
        self.type_id = 2

class Carnivore(Organism):
    """肉食動物"""
    def __init__(self, x, y, config, parent_ids=None, init_satiety_level=None):
        super().__init__(x, y, config, parent_ids, init_satiety_level=init_satiety_level)
        self.type_id = 3