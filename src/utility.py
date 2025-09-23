import torch

def get_observation_tensor(grid, x, y, obs_range, device):
    """
    指定された座標(x, y)から観測範囲内の環境情報を切り出し、テンソルとして返す。
    範囲外は0でパディングする。
    
    Args:
        grid (torch.Tensor): 環境全体を表すグリッド
        x (int): 中心のx座標
        y (int): 中心のy座標
        obs_range (int): 観測範囲（マンハッタン距離）
        device (torch.device): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: 観測範囲のテンソル (channels, height, width)
    """
    _, H, W = grid.shape
    size = 2 * obs_range + 1
    
    # パディング付きのグリッドを作成
    padded_grid = torch.nn.functional.pad(grid, (obs_range, obs_range, obs_range, obs_range))
    
    # 目的の範囲を切り出す
    observation = padded_grid[:, y:y+size, x:x+size]
    return observation.to(device)


def get_action_mask(grid, x, y, device):
    """
    指定された座標で実行不可能な行動をマスクするテンソルを生成する。
    
    Args:
        grid (torch.Tensor): 環境全体を表すグリッド
        x (int): 現在のx座標
        y (int): 現在のy座標
        device (torch.device): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: 行動マスク (1=実行可能, 0=実行不可能)
    """
    _, H, W = grid.shape
    mask = torch.ones(6, device=device) # 6つの行動すべてを可能として初期化
    
    # 境界チェック
    if y == 0: mask[0] = 0 # 上へ移動不可
    if y == H - 1: mask[1] = 0 # 下へ移動不可
    if x == 0: mask[2] = 0 # 左へ移動不可
    if x == W - 1: mask[3] = 0 # 右へ移動不可
        
    # TODO: 生殖マスク（周囲に空きがない場合など）のロジックを追加
    
    return mask