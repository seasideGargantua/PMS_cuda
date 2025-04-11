import torch
from torch import Tensor
import torch.nn.functional as F
from auxilary import PMSSetting, DisparityPlane

class Propagation:
    def __init__(self, width: int, height: int, setting: PMSSetting, cost_left: Tensor, cost_right: Tensor):
        self.setting = setting

    def propagate(self, disparity, plane, norm, image):
        # Placeholder for the propagation logic
        pass

    def compute_cost(self, image_left: Tensor, image_right: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the cost volume for left and right images."""
        assert len(image_left.shape) == 3 and len(image_right.shape) == 3, "Input images must be 3D tensors"
        assert image_left.shape == image_right.shape, "Left and right images must have the same shape"

        # Compute the cost volume
        cost_left = self.compute_cost_volume(image_left)
        cost_right = self.compute_cost_volume(image_right)

        return cost_left, cost_right

def Propagation(direction: int, width: int, height: int, is_fource_fpw: bool=False) -> Tuple[Tensor, Tensor]:
    """
    Propagate the disparity and plane information from one view to another.
    
    Args:
        dir (int): The direction of propagation.
        width (int): The width of the images.
        height (int): The height of the images.
        setting (PMSSetting): The settings for the propagation.
        cost_left (Tensor): The cost volume for the left image.
        cost_right (Tensor): The cost volume for the right image.

    Returns:
        Tuple[Tensor, Tensor]: The propagated disparity and plane information.
    """
    x = direction > 0 ? 0 : width-1
    for i in range(width):
        y = direction > 0 ? 0 : height-1
        for j in range(height):
            # Compute the cost volume for the current pixel
            SpatialPropagation(x, y, direction)

            if not is_fource_fpw:
				PlaneRefine(x, y)

            ViewPropagation(x, y)
    
            y += direction
        x += direction

    return disparity

def SpatialPropagation(direction: int, x: int, y: int, 
                        width:int, height: int, 
                        plane_left: DisparityPlane, plane_right: DisparityPlane, 
                        cost_left: Tensor, cost_right: Tensor) -> None:
    """
    Propagate the disparity and plane information in the spatial domain.
    
    Args:
        direction (int): The direction of propagation.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        width (int): The width of the images.
        height (int): The height of the images.
        plane_left (DisparityPlane): The plane information for the left image.
        plane_right (DisparityPlane): The plane information for the right image.
        cost_left (Tensor): The cost volume for the left image.
        cost_right (Tensor): The cost volume for the right image.
    """
    # Placeholder for the spatial propagation logic
    xd = x-direction
    if xd >= 0 and xd < width:
        if plane_left[y, xd] != plane_left[y, x]:
            cost = compute_cost(x, y, plane_left)
            if cost < cost_left[y, x]:
				plane_p = plane
				cost_p = cost

    yd = y-direction
    if yd >= 0 and yd < height:
        if plane_left[yd, x] != plane_left[y, x]:
            cost = compute_cost(x, y, plane_left)
            if cost < cost_left[y, x]:
                plane_p = plane
                cost_p = cost

def ViewPropagation(direction: int, x: int, y: int, 
                        width:int, height: int, 
                        plane_left: DisparityPlane, plane_right: DisparityPlane, 
                        cost_left: Tensor, cost_right: Tensor) -> None:
    """
    Propagate the disparity and plane information in the view domain.
    
    Args:
        direction (int): The direction of propagation.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        width (int): The width of the images.
        height (int): The height of the images.
        plane_left (DisparityPlane): The plane information for the left image.
        plane_right (DisparityPlane): The plane information for the right image.
        cost_left (Tensor): The cost volume for the left image.
        cost_right (Tensor): The cost volume for the right image.
    """
    # Placeholder for the view propagation logic
    # 计算左视图中像素 p 的索引
    plane_p = plane_left.get_plane(x, y)

    # 计算 p 的视差
    d_p = plane_p.to_disparity(x, y)

    # 计算右视图中对应像素的 x 坐标
    xr = round(x - d_p)
    if xr < 0 or xr >= self.width:
        return

    # 计算右视图中像素 q 的索引
    q = y * self.width + xr
    plane_q = self.plane_right[q]
    cost_q = self.cost_right[q]

    # 将左视图的平面参数转换到右视图
    plane_p2q = plane_p.to_another_view(x, y)
    d_q = plane_p2q.to_disparity(xr, y)

    # 计算新的代价值
    cost = self.cost_cpt.compute_a(xr, y, plane_p2q)

    # 如果新的代价值更小，则更新右视图的平面参数和代价值
    if cost < cost_q:
        self.plane_right[q] = plane_p2q
        self.cost_right[q] = cost

def PlaneRefine(x: int, y: int, 
                width:int, height: int, 
                plane_left: DisparityPlane, plane_right: DisparityPlane, 
                cost_left: Tensor, cost_right: Tensor) -> None:
    """
    Refine the plane information using the left and right images.
    
    Args:
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        width (int): The width of the images.
        height (int): The height of the images.
        plane_left (DisparityPlane): The plane information for the left image.
        plane_right (DisparityPlane): The plane information for the right image.
        cost_left (Tensor): The cost volume for the left image.
        cost_right (Tensor): The cost volume for the right image.
    """
    # Placeholder for the plane refinement logic
    max_disp = float(self.option.max_disparity)
    min_disp = float(self.option.min_disparity)

    # 随机数生成器
    rand_d = lambda: random.uniform(-1, 1)
    rand_n = lambda: random.uniform(-1, 1)

    # 获取当前像素的平面、代价和视差
    p = y * width + x
    plane_p = self.plane_left[p]
    cost_p = self.cost_left[p]

    d_p = plane_p.to_disparity(x, y)
    norm_p = plane_p.to_normal()

    disp_update = (max_disp - min_disp) / 2.0
    norm_update = 1.0
    stop_thres = 0.1

    # 优化循环
    while disp_update > stop_thres:
        # 在 -disp_update ~ disp_update 范围内生成一个随机视差增量
        disp_rd = rand_d() * disp_update
        if self.option.is_integer_disp:
            disp_rd = round(disp_rd)

        # 更新视差
        d_p_new = d_p + disp_rd
        if d_p_new < min_disp or d_p_new > max_disp:
            disp_update /= 2
            norm_update /= 2
            continue

        # 在 -norm_update ~ norm_update 范围内生成随机法向量增量
        if not self.option.is_fource_fpw:
            norm_rd = torch.tensor([
                rand_n() * norm_update,
                rand_n() * norm_update,
                rand_n() * norm_update
            ], dtype=torch.float32)

            # 确保 z 不为 0
            while norm_rd[2] == 0.0:
                norm_rd[2] = rand_n() * norm_update
        else:
            norm_rd = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # 更新法向量
        norm_p_new = norm_p + norm_rd
        norm_p_new = norm_p_new / torch.norm(norm_p_new)  # 归一化

        # 创建新的视差平面
        plane_new = DisparityPlane(x, y, norm_p_new, d_p_new)

        # 计算新的代价值
        if plane_new != plane_p:
            cost = self.cost_cpt.compute_a(x, y, plane_new)

            if cost < cost_p:
                plane_p = plane_new
                cost_p = cost
                d_p = d_p_new
                norm_p = norm_p_new

        disp_update /= 2.0
        norm_update /= 2.0