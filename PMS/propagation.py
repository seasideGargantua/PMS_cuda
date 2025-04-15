import torch
from torch import Tensor
import torch.nn.functional as F
from .auxilary import PMSSetting
import PMS.cuda as _C

def propagation(num_iter: int,
                setting: PMSSetting,
                img_left: Tensor,
                img_right: Tensor,
                grad_left: Tensor,
                grad_right: Tensor,
                plane_left: Tensor,
                plane_right: Tensor,
                cost_left: Tensor,
                cost_right: Tensor,
                rand_d: Tensor,
                rand_n: Tensor,
                left_img: bool = True):
    """Plane and Cost Propagation."""
    if left_img:
        return _Propagation.apply(
            num_iter,
            setting.patch_size,
            setting.gamma,
            setting.alpha,
            setting.tau_col,
            setting.tau_grad,
            setting.min_disparity,
            setting.max_disparity,
            setting.is_integer_disp,
            setting.is_fource_fpw,
            img_left,
            img_right,
            grad_left,
            grad_right,
            plane_left,
            plane_right,
            cost_left,
            cost_right,
            rand_d,
            rand_n
        )
    else:
        return _Propagation.apply(
            num_iter,
            setting.patch_size,
            setting.gamma,
            setting.alpha,
            setting.tau_col,
            setting.tau_grad,
            -setting.max_disparity,
            -setting.min_disparity,
            setting.is_integer_disp,
            setting.is_fource_fpw,
            img_right,
            img_left,
            grad_right,
            grad_left,
            plane_right,
            plane_left,
            cost_right,
            cost_left,
            rand_d,
            rand_n
        )

class _Propagation(torch.autograd.Function):
    """Plane and Cost Propagation."""
    
    @staticmethod
    def forward(
        ctx,
        num_iter,
        patch_size,
        gamma,
        alpha,
        tau_col,
        tau_grad,
        min_disp,
        max_disp,
        is_integer_disp,
        is_fource_fpw,
        img_left,
        img_right,
        grad_left,
        grad_right,
        plane_left,
        plane_right,
        cost_left,
        cost_right,
        rand_d,
        rand_n
    ):

        plane_left, plane_right, cost_left, cost_right = _C.propagation(
            num_iter,
            patch_size,
            gamma,
            alpha,
            tau_col,
            tau_grad,
            min_disp,
            max_disp,
            is_integer_disp,
            is_fource_fpw,
            img_left,
            img_right,
            grad_left,
            grad_right,
            plane_left,
            plane_right,
            cost_left,
            cost_right,
            rand_d,
            rand_n
        )

        return plane_left, plane_right, cost_left, cost_right
