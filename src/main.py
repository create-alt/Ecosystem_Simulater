import matplotlib.pyplot as plt
from config import Config
from environment import Env
from agent import Agent
from visualizer import Visualizer

def main():
    config = Config()
    env = Env(config)
    
    herbivore_agent = Agent(input_channels=4, action_num=6, config=config)
    carnivore_agent = Agent(input_channels=4, action_num=6, config=config)

    env.reset()

    # ==================================================================
    # 1. シミュレーション実行と履歴の保存
    # ==================================================================
    print(f"Running simulation for {config.SIMULATION_STEPS} steps on {config.DEVICE}...")
    history = []
    for step in range(config.SIMULATION_STEPS):
        # --- 行動決定 ---
        if env.herbivores:
            herb_obs = env.get_batch_observations(env.herbivores)
            herb_masks = env.get_batch_masks(env.herbivores)
            herb_actions = herbivore_agent.select_actions(herb_obs, herb_masks)
        else: herb_actions = []

        if env.carnivores:
            carn_obs = env.get_batch_observations(env.carnivores)
            carn_masks = env.get_batch_masks(env.carnivores)
            carn_actions = carnivore_agent.select_actions(carn_obs, carn_masks)
        else: carn_actions = []

        # --- 環境ステップと学習 ---
        rewards = env.step(herb_actions, carn_actions)
        
        if rewards["herbivore"]:
            avg_herb_reward = sum(rewards["herbivore"].values()) / len(rewards["herbivore"])
            herbivore_agent.rewards.append(avg_herb_reward)
            herbivore_agent.optimize_model()

        if rewards["carnivore"]:
            avg_carn_reward = sum(rewards["carnivore"].values()) / len(rewards["carnivore"])
            carnivore_agent.rewards.append(avg_carn_reward)
            carnivore_agent.optimize_model()

        # --- 状態を履歴に保存 ---
        # GPUメモリ上のテンソルはCPUに移動させてから保存
        history.append({
            'grid': env.grid.cpu(),
            'herb_count': len(env.herbivores),
            'carn_count': len(env.carnivores)
        })
        
        if step % 100 == 0:
            print(f"  ... Step {step} completed.")

    print("Simulation finished.")

    # ==================================================================
    # 2. インタラクティブな可視化
    # ==================================================================
    if not history:
        print("No history to visualize.")
        return
        
    print("Starting visualizer... (Use <- -> arrows to navigate, 'q' or 'escape' to quit)")
    visualizer = Visualizer(config.ENV_SIZE)
    current_step = 0
    
    # --- キー操作の定義 ---
    def on_key_press(event):
        nonlocal current_step
        if event.key == 'right':
            current_step = min(len(history) - 1, current_step + 1)
        elif event.key == 'left':
            current_step = max(0, current_step - 1)
        elif event.key in ['q', 'escape']:
            plt.close(visualizer.fig) # ウィンドウを閉じる
            return
        
        # 画面更新
        state = history[current_step]
        visualizer.update(state['grid'], current_step, state['herb_count'], state['carn_count'])

    # --- イベントハンドラを登録 ---
    visualizer.fig.canvas.mpl_connect('key_press_event', on_key_press)

    # --- 初期画面の表示 ---
    state = history[0]
    visualizer.update(state['grid'], 0, state['herb_count'], state['carn_count'])
    
    # --- ウィンドウ表示 (ユーザーが閉じるまでここでブロック) ---
    visualizer.show()
    
    print("Visualizer closed.")

    # --- モデルの保存 ---
    herbivore_agent.save_model("herbivore_model.pth")
    carnivore_agent.save_model("carnivore_model.pth")
    print("Models saved.")

if __name__ == '__main__':
    main()