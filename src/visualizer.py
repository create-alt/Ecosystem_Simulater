import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, env_size):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(np.zeros((env_size, env_size, 3)))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.manager.set_window_title('Ecosystem Simulator')
        
    def _grid_to_rgb(self, grid):
        """
        環境グリッド(one-hot)をRGB画像に変換する
        0:空(黒), 1:植物(緑), 2:草食(青), 3:肉食(赤)
        """
        # CPUにあることを想定 (mainで .cpu() 済み)
        grid_np = grid.numpy()
        grid_np = np.transpose(grid_np, (1, 2, 0))
        
        rgb_image = np.zeros((*grid_np.shape[:2], 3))
        
        rgb_image[grid_np[:, :, 1] == 1] = [0, 1, 0]  # 植物 -> 緑
        rgb_image[grid_np[:, :, 2] == 1] = [0, 0, 1]  # 草食 -> 青
        rgb_image[grid_np[:, :, 3] == 1] = [1, 0, 0]  # 肉食 -> 赤

        return rgb_image

    def update(self, grid, step, herb_count, carn_count):
        """表示を更新する"""
        rgb_grid = self._grid_to_rgb(grid)
        self.im.set_data(rgb_grid)
        self.ax.set_title(f"Step: {step} | Herbivores: {herb_count} | Carnivores: {carn_count}")
        # draw_idle() は再描画が必要なタイミングで効率的に描画する
        self.fig.canvas.draw_idle()
        
    def show(self):
        """ウィンドウを表示し、ユーザーが閉じるまで待機する"""
        plt.show()