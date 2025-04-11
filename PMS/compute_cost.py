import torch
from torch import Tensor

import torch
import torch.nn.functional as F
import math

class CostComputer:
    def __init__(self, width, height, alpha, tau_col, tau_grad, gamma, min_disp, max_disp, patch_size, cost_punish):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.tau_col = tau_col
        self.tau_grad = tau_grad
        self.gamma = gamma
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.patch_size = patch_size
        self.cost_punish = cost_punish

    def get_color(self, img, x, y):
        # 双线性插值获取颜色值
        x0 = torch.floor(x).long()
        x1 = torch.ceil(x).long()
        y0 = torch.floor(y).long()
        y1 = torch.ceil(y).long()

        x0 = torch.clamp(x0, 0, self.width - 1)
        x1 = torch.clamp(x1, 0, self.width - 1)
        y0 = torch.clamp(y0, 0, self.height - 1)
        y1 = torch.clamp(y1, 0, self.height - 1)

        q00 = img[y0, x0]
        q01 = img[y0, x1]
        q10 = img[y1, x0]
        q11 = img[y1, x1]

        wx = x - x0
        wy = y - y0

        return (1 - wx) * (1 - wy) * q00 + wx * (1 - wy) * q01 + (1 - wx) * wy * q10 + wx * wy * q11

    def get_gradient(self, grad, x, y):
        # 双线性插值获取梯度值
        return self.get_color(grad, x, y)

    def compute(self, img_right, grad_right, col_p, grad_p, x, y, d):
        xr = x - d
        if xr < 0.0 or xr >= self.width:
            return (1 - self.alpha) * self.tau_col + self.alpha * self.tau_grad

        # 颜色空间距离
        col_q = self.get_color(img_right, xr, y)
        dc = torch.min(torch.abs(col_p - col_q).sum(), torch.tensor(self.tau_col))

        # 梯度空间距离
        grad_q = self.get_gradient(grad_right, xr, y)
        dg = torch.min(torch.abs(grad_p - grad_q).sum(), torch.tensor(self.tau_grad))

        # 综合代价
        return (1 - self.alpha) * dc + self.alpha * dg

    def compute_a(self, img_left, grad_left, img_right, grad_right, x, y, disparity_plane):
        pat = self.patch_size // 2
        col_p = self.get_color(img_left, x, y)
        cost = 0.0

        for r in range(-pat, pat + 1):
            yr = y + r
            for c in range(-pat, pat + 1):
                xc = x + c
                if yr < 0 or yr >= self.height or xc < 0 or xc >= self.width:
                    continue

                # 计算视差值
                d = disparity_plane.to_disparity(xc, yr)
                if d < self.min_disp or d > self.max_disp:
                    cost += self.cost_punish
                    continue

                # 计算权值
                col_q = self.get_color(img_left, xc, yr)
                dc = torch.abs(col_p - col_q).sum()
                w = torch.exp(-dc / self.gamma)

                # 累加代价
                grad_q = self.get_gradient(grad_left, xc, yr)
                cost += w * self.compute(img_right, grad_right, col_q, grad_q, xc, yr, d)

        return cost
