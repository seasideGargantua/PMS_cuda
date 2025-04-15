import numpy as np
import torch
from PMS.auxilary import PMSSetting
from PMS.match import match
from PIL import Image

def main():
    left_img_path = "./data/Cone/im2.png"
    right_img_path = "./data/Cone/im6.png"
    left_img = np.array(Image.open(left_img_path))
    right_img = np.array(Image.open(right_img_path))
    left_img = torch.from_numpy(left_img).float().cuda()
    right_img = torch.from_numpy(right_img).float().cuda()

    setting = PMSSetting(
        patch_size=35,
        min_disparity=0,
        max_disparity=64,
        gamma=10.0,
        alpha=0.9,
        tau_col=10.0,
        tau_grad=2.0,
        num_iters=3,
        is_check_lr=False,
        lrcheck_thres=0,
        is_fill_holes=False,
        is_fource_fpw=False,
        is_integer_disp=False
    )

    print("Start matching...")
    left_disp, right_disp = match(left_img, right_img, setting)

    left_disp = left_disp.cpu().numpy()
    right_disp = right_disp.cpu().numpy()
    left_disp = np.where(abs(left_disp) > 1024, 0, left_disp)
    right_disp = np.where(abs(right_disp) > 1024, 0, right_disp)
    left_disp = (left_disp - left_disp.min()) / (left_disp.max()-left_disp.min()) * 255
    right_disp = (right_disp - right_disp.min()) / (right_disp.max()-right_disp.min()) * 255
    
    # save the disparity maps
    print("Saving disparity maps...")
    left_disp_img = Image.fromarray(left_disp.astype(np.uint8))
    right_disp_img = Image.fromarray(right_disp.astype(np.uint8))
    left_disp_img.save("left_disp.png")
    right_disp_img.save("right_disp.png")

if __name__ == "__main__":
    main()