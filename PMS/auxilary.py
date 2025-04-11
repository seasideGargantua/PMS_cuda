from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

class PMSSetting:
    def __init__(self, patch_size: int=35, min_disparity: int=0, max_disparity: int=64,
                    gamma: float=10.0, alpha: float=0.9, tau_col: float=10.0, tau_grad: float=2.0,
                    num_iters: int=3, is_check_lr: bool=False, lrcheck_thres: float=0,
                    is_fill_holes: bool=False, is_fource_fpw: bool=False, is_integer_disp: bool=False):
        self.patch_size = patch_size
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.min_disparity_r = -min_disparity   # for propagation
        self.max_disparity_r = -max_disparity   # for propagation
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

class DisparityPlane:
    def __init__(self, width: int, height: int, disp: Tensor=None, norm: Tensor=None, plane: Tensor=None):
        self.width = width
        self.height = height
        assert width > 0 and height > 0, "Width and height must be positive integers"
        assert (disp is not None and norm is not None) or plane is not None, "At least disp and norm, or plane must be provided"
        if plane is not None:
            self.plane = plane
        else:
            xy_plane = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height)), dim=-1).float()
            # (n.x * x + n.y * y + n.z * d) / n.z;
            xyd = torch.cat([xy_plane, disp.unsqueeze(-1)], dim=-1)
            norm = norm / norm[..., 2:3]
            plane_z = norm * xyd
            self.plane = torch.cat([-norm[..., :2], plane_z], dim=-1)
    
    def get_plane(self, px: int, py: int):
        return self.plane[py, px]

    def to_disparity(self, px: int, py: int, x: int, y: int):
        return self.planes[py, px] @ torch.tensor([x, y, 1.], device=self.plane.device).unsqueeze(-1)

    def to_normal(self, px: int, py: int):
        return torch.tensor([self.planes[py, px, 0], self.planes[py, px, 1], 1.]).normalize(dim=-1)

    def to_another_view(self, x: int, y: int):
        denom = 1. / (self.plane[y, x, 0] - 1.)
        return torch.tensor([self.plane[y, x, 0] * denom, self.plane[y, x, 1] * denom, self.plane[y, x, 2] * denom], device=self.plane.device)

def compute_gray(rgb: Tensor) -> Tensor:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

def compute_gradients(image: Tensor) -> Tuple[Tensor, Tensor]:
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