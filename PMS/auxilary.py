from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import PMS.cuda as _C

class PMSSetting:
    def __init__(self, patch_size: int=35, min_disparity: int=0, max_disparity: int=64,
                    gamma: float=10.0, alpha: float=0.9, tau_col: float=10.0, tau_grad: float=2.0,
                    num_iters: int=3, is_check_lr: bool=True, lrcheck_thres: float=1.0,
                    is_fill_holes: bool=False, is_fource_fpw: bool=False, is_integer_disp: bool=False):
        self.patch_size = patch_size
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.gamma = gamma
        self.alpha = alpha
        self.tau_col = tau_col
        self.tau_grad = tau_grad
        self.num_iters = num_iters
        self.is_check_lr = is_check_lr
        self.lrcheck_thres = lrcheck_thres
        self.is_fill_holes = is_fill_holes
        self.is_fource_fpw = is_fource_fpw
        self.is_integer_disp = is_integer_disp

def plane2disparity(plane: Tensor) -> Tensor:
    """Convert plane to disparity."""
    assert len(plane.shape) == 3 and plane.shape[-1] == 3, "Input plane must be a 3D tensor with last dimension of size 3"
    assert plane.shape[0] > 0 and plane.shape[1] > 0, "Input plane must have positive height and width"
    
    height, width = plane.shape[:2]
    xy_plane = torch.stack(torch.meshgrid(torch.arange(height, device=plane.device), torch.arange(width, device=plane.device)), dim=-1).float()
    # planes[py, px] @ torch.tensor([x, y, 1.], device=self.plane.device).unsqueeze(-1)
    xy = torch.cat([xy_plane, torch.ones((height, width, 1), device=plane.device)], dim=-1)
    return torch.einsum('ijk,ijk->ij', plane, xy)

def compute_gray(rgb: Tensor) -> Tensor:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

def compute_gradients(gray: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute gradients of the image."""
    assert len(gray.shape) == 2, "Input gray image must be a 2D tensor"
    
    height, width = gray.shape
    
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=gray.device)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=gray.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)  # (out_channels, in_channels, kernel_height, kernel_width)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    gray = gray.unsqueeze(0).unsqueeze(0)  # (batch_size=1, channels=1, height, width)
    
    grad_x = F.conv2d(gray, sobel_x, padding=1) / 8.0  # 保持尺寸一致
    grad_y = F.conv2d(gray, sobel_y, padding=1) / 8.0
    
    grad_x = grad_x.squeeze(0).squeeze(0)  # (height, width)
    grad_y = grad_y.squeeze(0).squeeze(0)
    
    grad = torch.stack((grad_x, grad_y), dim=-1)  # (height, width, 2)
    
    return grad

def initiate_cost(
    setting: PMSSetting,
    img_left: Tensor,
    img_right: Tensor,
    grad_left: Tensor,
    grad_right: Tensor,
    plane_left: Tensor,
    plane_right: Tensor
):
    """Initiate the cost volume."""
    return _InitiateCost.apply(
        setting.patch_size,
        setting.gamma,
        setting.alpha,
        setting.tau_col,
        setting.tau_grad,
        setting.min_disparity,
        setting.max_disparity,
        img_left,
        img_right,
        grad_left,
        grad_right,
        plane_left,
        plane_right
    )

class _InitiateCost(torch.autograd.Function):
    """ Initiate the Cost """

    @staticmethod
    def forward(
        ctx,
        patch_size,
        gamma,
        alpha,
        tau_col,
        tau_grad,
        min_disp,
        max_disp,
        img_left,
        img_right,
        grad_left,
        grad_right,
        plane_left,
        plane_right
    ):

        cost_left, cost_right = _C.initiate_cost(
            patch_size,
            gamma,
            alpha,
            tau_col,
            tau_grad,
            min_disp,
            max_disp,
            img_left,
            img_right,
            grad_left,
            grad_right,
            plane_left,
            plane_right
            )

        return cost_left, cost_right