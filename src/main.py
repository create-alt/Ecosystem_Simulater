import os
import torch
import matplotlib.pyplot as plt
from config import Config
from environment import Env
from agent import Agent
from visualizer import Visualizer

def plot_rewards(herb_rewards, carn_rewards):
    """
    シミュレーション終了後に報酬の履歴をグラフで可視化し、ファイルに保存する。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(herb_rewards, label='Herbivore Average Reward', color='blue', alpha=0.7)
    plt.plot(carn_rewards, label='Carnivore Average Reward', color='red', alpha=0.7)
    plt.title('Average Reward per Step during Training')
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('reward_graph.png')
    print("Reward graph saved as reward_graph.png")
    plt.show()

def train(herbivore_agent, carnivore_agent, config):
    """複数環境でエージェントを並列学習させる"""
    envs = [Env(config) for _ in range(config.NUM_ENVIRONMENTS)]
    for env in envs:
        env.reset()

    print(f"Training on {config.NUM_ENVIRONMENTS} environments for {config.TRAINING_STEPS} steps...")
    
    herbivore_reward_history = []
    carnivore_reward_history = []

    for step in range(config.TRAINING_STEPS):
        # 1. 全環境から観測とマスクを収集
        all_herb_obs, all_herb_masks = [], []
        all_carn_obs, all_carn_masks = [], []

        for env in envs:
            if env.herbivores:
                all_herb_obs.append(env.get_batch_observations(env.herbivores))
                all_herb_masks.append(env.get_batch_masks(env.herbivores))
            if env.carnivores:
                all_carn_obs.append(env.get_batch_observations(env.carnivores))
                all_carn_masks.append(env.get_batch_masks(env.carnivores))
        
        # 2. 行動をまとめて決定
        herb_actions_flat, carn_actions_flat = None, None
        if all_herb_obs:
            herb_obs_batch = torch.cat(all_herb_obs, dim=0)
            herb_masks_batch = torch.cat(all_herb_masks, dim=0)
            herb_actions_flat = herbivore_agent.select_actions(herb_obs_batch, herb_masks_batch)
        
        if all_carn_obs:
            carn_obs_batch = torch.cat(all_carn_obs, dim=0)
            carn_masks_batch = torch.cat(all_carn_masks, dim=0)
            carn_actions_flat = carnivore_agent.select_actions(carn_obs_batch, carn_masks_batch)

        # 3. 各環境でステップを進め、報酬を収集
        total_herb_rewards, total_carn_rewards = [], []
        
        # 行動を各環境に分配
        herb_idx_counter, carn_idx_counter = 0, 0
        for i, env in enumerate(envs):
            num_herbs = len(env.herbivores)
            num_carns = len(env.carnivores)
            
            herb_actions = herb_actions_flat[herb_idx_counter : herb_idx_counter + num_herbs] if herb_actions_flat is not None and num_herbs > 0 else []
            carn_actions = carn_actions_flat[carn_idx_counter : carn_idx_counter + num_carns] if carn_actions_flat is not None and num_carns > 0 else []
            
            herb_idx_counter += num_herbs
            carn_idx_counter += num_carns

            rewards = env.step(herb_actions, carn_actions)
            if rewards["herbivore"]:
                total_herb_rewards.extend(rewards["herbivore"].values())
            if rewards["carnivore"]:
                total_carn_rewards.extend(rewards["carnivore"].values())

        # 4. 平均報酬を記録
        avg_herb_reward = sum(total_herb_rewards) / len(total_herb_rewards) if total_herb_rewards else 0
        herbivore_agent.rewards.append(avg_herb_reward)
        herbivore_reward_history.append(avg_herb_reward)

        avg_carn_reward = sum(total_carn_rewards) / len(total_carn_rewards) if total_carn_rewards else 0
        carnivore_agent.rewards.append(avg_carn_reward)
        carnivore_reward_history.append(avg_carn_reward)
        
        # 5. 一定間隔でモデルを更新
        if (step + 1) % config.UPDATE_INTERVAL == 0:
            herbivore_agent.optimize_model()
            carnivore_agent.optimize_model()
            
        if step % 100 == 0:
            print(f"  ... Training Step {step} | Avg Herb Reward: {avg_herb_reward:.3f} | Avg Carn Reward: {avg_carn_reward:.3f}")
            
    print("Training finished.")
    return herbivore_reward_history, carnivore_reward_history

def visualize_simulation(herbivore_agent, carnivore_agent, config):
    """学習済みエージェントを使ってシミュレーションを可視化する"""
    print(f"\nRunning visualization simulation for {config.VISUALIZATION_STEPS} steps...")
    
    env = Env(config)
    env.reset()
    history = []

    for step in range(config.VISUALIZATION_STEPS):
        # --- 行動決定 ---
        # 可視化中は学習しないため、エージェントの内部ログは使用しない
        with torch.no_grad():
            if env.herbivores:
                herb_obs = env.get_batch_observations(env.herbivores)
                herb_masks = env.get_batch_masks(env.herbivores)
                # policyから直接行動を選択（ログは保存しない）
                policy, _ = herbivore_agent.model(herb_obs)
                policy *= herb_masks
                if policy.sum(dim=1).min() == 0:
                    bad_indices = (policy.sum(dim=1) == 0)
                    policy_raw, _ = herbivore_agent.model(herb_obs[bad_indices])
                    policy[bad_indices] = policy_raw
                herb_actions = torch.distributions.Categorical(policy).sample()
            else:
                herb_actions = []

            if env.carnivores:
                carn_obs = env.get_batch_observations(env.carnivores)
                carn_masks = env.get_batch_masks(env.carnivores)
                policy, _ = carnivore_agent.model(carn_obs)
                policy *= carn_masks
                if policy.sum(dim=1).min() == 0:
                    bad_indices = (policy.sum(dim=1) == 0)
                    policy_raw, _ = carnivore_agent.model(carn_obs[bad_indices])
                    policy[bad_indices] = policy_raw
                carn_actions = torch.distributions.Categorical(policy).sample()
            else:
                carn_actions = []

        # --- 環境ステップ ---
        env.step(herb_actions, carn_actions)
        
        # --- 状態を履歴に保存 ---
        history.append({
            'grid': env.grid.cpu(),
            'herb_count': len(env.herbivores),
            'carn_count': len(env.carnivores)
        })
        
        if step % 100 == 0:
            print(f"  ... Visualization Step {step} completed.")

    # --- インタラクティブな可視化 ---
    if not history:
        print("No history to visualize.")
        return
        
    print("Starting visualizer... (Use <- -> arrows to navigate, 'q' or 'escape' to quit)")
    visualizer = Visualizer(config.ENV_SIZE)
    current_step = 0
    
    def on_key_press(event):
        nonlocal current_step
        if event.key == 'right':
            current_step = min(len(history) - 1, current_step + 1)
        elif event.key == 'left':
            current_step = max(0, current_step - 1)
        elif event.key in ['q', 'escape']:
            plt.close(visualizer.fig)
            return
        
        state = history[current_step]
        visualizer.update(state['grid'], current_step, state['herb_count'], state['carn_count'])

    visualizer.fig.canvas.mpl_connect('key_press_event', on_key_press)
    state = history[0]
    visualizer.update(state['grid'], 0, state['herb_count'], state['carn_count'])
    visualizer.show()
    print("Visualizer closed.")

def main():
    config = Config()
    
    # --- エージェントとモデルの初期化 ---
    herbivore_agent = Agent(input_channels=4, action_num=6, config=config)
    carnivore_agent = Agent(input_channels=4, action_num=6, config=config)

    herbivore_model_path = "herbivore_model.pth"
    carnivore_model_path = "carnivore_model.pth"

    # 学習済みモデルの読み込み
    if os.path.exists(herbivore_model_path):
        try:
            herbivore_agent.load_model(herbivore_model_path)
            print(f"Loaded pre-trained model for herbivores from: {herbivore_model_path}")
        except Exception as e:
            print(f"Error loading herbivore model: {e}. Starting with a new model.")
    else:
        print("No pre-trained model found for herbivores. Starting with a new model.")

    if os.path.exists(carnivore_model_path):
        try:
            carnivore_agent.load_model(carnivore_model_path)
            print(f"Loaded pre-trained model for carnivores from: {carnivore_model_path}")
        except Exception as e:
            print(f"Error loading carnivore model: {e}. Starting with a new model.")
    else:
        print("No pre-trained model found for carnivores. Starting with a new model.")
    
    # ==================================================================
    # 1. 複数環境で学習
    # ==================================================================
    herb_rewards, carn_rewards = train(herbivore_agent, carnivore_agent, config)

    # ==================================================================
    # 2. モデルの保存
    # ==================================================================
    herbivore_agent.save_model(herbivore_model_path)
    carnivore_agent.save_model(carnivore_model_path)
    print("Models saved.")

    # ==================================================================
    # 3. 学習曲線のプロット
    # ==================================================================
    plot_rewards(herb_rewards, carn_rewards)
    
    # ==================================================================
    # 4. 学習済みモデルでシミュレーションを可視化
    # ==================================================================
    visualize_simulation(herbivore_agent, carnivore_agent, config)

if __name__ == '__main__':
    main()