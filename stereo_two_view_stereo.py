import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d
import sys

from stereo_dataloader import load_middlebury_data
from stereo_utils import viz_camera_poses

def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max

def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w

    Returns
    -------
    i_R_j, i_T_j: [3,3], [3,1],
        p_i = i_R_j @ p_j + i_T_j
    B: float,
        the baseline
    """

    i_R_j = i_R_w @ j_R_w.T
    i_T_j = i_T_w - i_R_w @ j_R_w.T @ j_T_w
    B = np.linalg.norm(i_T_j)

    return i_R_j, i_T_j, B

def rectify_2view(rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    rgb_i_rect, rgb_j_rect: [H,W,3],[H,W,3],
        the rectified images.
    K_i_corr, K_j_corr: [3,3],[3,3],
        the corrected camera projection matrix. 
    
    K_i_corr and K_j_corr are already corrected for padding.
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "Input images must be the same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ rect_R_i @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ rect_R_j @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "Assumes both cameras share the same principal point cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    H_i = K_i_corr @ rect_R_i @ np.linalg.inv(K_i)
    H_j = K_j_corr @ rect_R_j @ np.linalg.inv(K_j)

    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))
    

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr

def compute_rectification_R(i_T_j):
    """Compute the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    p_rect: [3,3],
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    EPS = 1e-8 
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS)

    r2 = e_i / np.linalg.norm(e_i)
    r1 = np.cross(e_i, [0, 0, 1])
    r3 = np.cross(r1, r2)
    rect_R_i = np.vstack((r1, r2, r3))
                                 

    return rect_R_i

def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    ssd: [M,N],
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    M = src.shape[0]
    N = dst.shape[0]
    ssd = np.zeros((M, N))
    for p in range(src.shape[0]):
        patch_left = src[p, :, :]
        error = (dst - patch_left) ** 2
        s = np.sum(error, axis=-1)
        s = np.sum(s, axis=-1)
        ssd[p, :] = s
                                     

    return ssd  # M,N

def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    sad: [M,N],
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    M = src.shape[0]
    N = dst.shape[0]
    sad = np.zeros((M, N))
    for p in range(src.shape[0]):
        patch_left = src[p, :, :]
        error = abs(dst - patch_left)
        s = np.sum(error, axis=-1)
        s = np.sum(s, axis=-1)
        sad[p, :] = s  
    
                                
    return sad  # M,N

def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    zncc: [M,N],
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    EPS = 1e-8 
    M = src.shape[0]
    N = dst.shape[0]
    zncc = np.zeros((src.shape[0], dst.shape[0]))
    for p in range(src.shape[0]):
        patch_left = src[p, :, :]
        patch_mean = np.mean(patch_left, axis=0)
        patch_left_mean = patch_left - patch_mean
        sigma_left_arr = np.std(patch_left, axis=0).reshape((1, src.shape[2]))
        sigma_left_arr[np.where(sigma_left_arr == 0)] = 1e-8

        patch_right_m = np.mean(dst, axis=1).reshape((dst.shape[0], 1, dst.shape[2]))
        patch_right_mean = dst - patch_right_m
        patch_right_std = np.std(dst, axis=1)
        final_sigma = (patch_right_std * sigma_left_arr).reshape((dst.shape[0], 1, dst.shape[2]))
        final_sigma[np.where(final_sigma == 0)] = 1e-8

        mul = ((patch_right_mean * patch_left_mean) / final_sigma).sum(axis=-1).sum(axis=-1)
        zncc[p, :] = mul 

    return zncc * (-1.0)  # M,N
          
def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    patch_buffer: [H, W, k_size**2, 3],
        The patch buffer for each pixel
    """

    pad_w = int(k_size / 2)
    patches = np.zeros((image.shape[0], image.shape[1], k_size**2, 3))
    for img in range(image.shape[2]):
        aug_img = np.pad(image[:, :, img], pad_w)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                x, y = row, col
                aug_x, aug_y = x + pad_w, y + pad_w
                patches[row, col, :, img] = aug_img[max(0, aug_x - pad_w):aug_x + pad_w + 1, max(0, aug_y - pad_w):aug_y + pad_w + 1].flatten()
    patch_buffer = patches

    return patch_buffer  # H,W,K**2,3
               
def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : the bias term of the disparity caused by different K matrices
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    disp_map = np.zeros(rgb_i.shape[:2], dtype=np.float64)
    lr_consistency_mask = np.ones_like(disp_map, dtype=np.float64)
    
    # 1. Compute the patch buffers for both images
    # NOTE: when computing patches, please use the syntax:
    # patch_buffer = img2patch_func(image, k_size)
    
    # 2. Compute the disparity and LR consistency for each pixel
    # 3. Compute disparity candidates

    h, w = rgb_i.shape[0], rgb_i.shape[1]
    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0

    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]

    rows, cols = h, w #v,u

    disp_map = np.zeros((rows, cols), dtype=np.float64) 
    lr_consistency_mask = np.zeros((rows, cols), dtype=np.float64) 

    for bla in range(cols):
        buf_i, buf_j = patches_i[:, bla], patches_j[:, bla] #get patch
        value = ssd_kernel(buf_i, buf_j)  
        #arr = []
        for v in range(value.shape[0]): #taking one row at a time
            best_matched_right_pixel = np.argmin(value[v,:]) #column index at which min is there
            #arr.append(disp_candidates[v,best_matched_right_pixel]) #appending the corresponding value
            disp_map[v,bla] = disp_candidates[v,best_matched_right_pixel]
            best_matched_left_pixel = value[:, best_matched_right_pixel].argmin() #this should match v
            consistent_flag = int(best_matched_left_pixel == v)
            lr_consistency_mask[v,bla] = consistent_flag 
            
        # np.array([arr.append(disp_candidates[k,min_elements[k]]) for k in range(min_elements.shape[0])])

    return disp_map, lr_consistency_mask

def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    dep_map: [H,W]
        depth map
    xyz_cam: [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame.
        You may find using np.meshgrid helpful here.
    """

    f_x = K[0][0] #fx maybe check
    f_y = K[1][1] #fx maybe check
    val = B*f_y
    d_invert = np.reciprocal(disp_map)
    dep_map = val*d_invert
    #print(disp_map[0][0], d_invert[0][0])
    
    #K_inv = np.linalg.inv(K)
    img_cent_x = K[0][-1]
    img_cent_y = K[1][-1]
    xyz_cam = np.zeros((disp_map.shape[0], disp_map.shape[1], 3))
    for row in range(disp_map.shape[0]):
        for col in range(disp_map.shape[1]):
            depth = dep_map[row][col]
            y_coord, x_coord = (row- img_cent_y)*depth/f_y, (col - img_cent_x)*depth/f_x
            z_coord = depth
            xyz_cam[row,col,:] = np.array([x_coord, y_coord, z_coord])         
                                

    return dep_map, xyz_cam

def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    
    NOTE: This can be done with one line of code!
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    pcl_world = c_R_w.T @ pcl_cam.T - (c_R_w.T @ c_T_w)
    pcl_world = pcl_world.T
                        

    return mask, pcl_world, pcl_cam, pcl_color

def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]  # p_i = i_R_w @ p_w + i_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]  # p_j = j_R_w @ p_w + j_T_w

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert i_T_j[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "Assumes both cameras share the same focal length in Y"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "Assumes both cameras share the same K on the X dimension"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "Rectified views are expected to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map

def main():
    DATA = load_middlebury_data("stereo_data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return

if __name__ == "__main__":
    main()
