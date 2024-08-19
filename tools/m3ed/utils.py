import numpy as np
import open3d as o3
import torch
import math
import cv2
import sys
import bisect
import visibility
from camera_model import CameraModel

def load_data(data, sensor='ovc', camera='rgb'):
    '''
        load data from hdf5 file
    '''
    if sensor not in ['ovc', 'prophesee', 'ouster']:
        raise "The sensor doesn't exist."

    if sensor == 'ovc':
        if camera not in ['left', 'right', 'rgb']:
            raise "The camera does not exist."
        
        data = data['ovc']
        ts = data['ts']                                                     # 1424 (interval=40000us)
        ts_map_prophesee_left_t = data['ts_map_prophesee_left_t']           # 1424 index
        data = data[camera]
        img = data['data']                                                  # (1424 800 1280 3)
        calib = data['calib']
        T_to_prophesee_left = np.asarray(calib['T_to_prophesee_left'])      # 4 4
        distortion_coeffs = np.asarray(calib['distortion_coeffs'])          # 4
        intrinsics = np.asarray(calib['intrinsics'])                        # 4
        resolution = np.asarray(calib['resolution'])                        # (1424 800)
        sample = {"img": img, "ts": ts, "ts_map_prophesee_left_t": ts_map_prophesee_left_t,
                "T_to_prophesee_left": T_to_prophesee_left, "distortion_coeffs": distortion_coeffs,
                "intrinsics": intrinsics, "resolution": resolution}
    
    elif sensor == 'prophesee':
        if camera not in ['left', 'right']:
            raise "The camera does not exist."
        
        data = data['prophesee']
        # print(data['left']['t'][0], data['left']['t'][-1])
        # print(data['right']['t'][0], data['right']['t'][-1])
        data = data[camera]
        calib = data['calib']
        distortion_coeffs = np.asarray(calib['distortion_coeffs'])      # 4
        intrinsics = np.asarray(calib['intrinsics'])                    # 4
        resolution = np.asarray(calib['resolution'])                    # (1280 720)
        ms_map_idx = data['ms_map_idx'][:]
        p, t, x, y = data['p'], data['t'], data['x'], data['y']
        T_to_prophesee_left = np.asarray(calib['T_to_prophesee_left'])

        sample = {"p": p, "t": t, "x": x, "y": y,
                  "distortion_coeffs": distortion_coeffs, "intrinsics": intrinsics, 'T_to_prophesee_left': T_to_prophesee_left,
                  "resolution": resolution, "ms_map_idx": ms_map_idx}
    
    return sample

def find_near_index(t_cur, ts, time_window=100000):
    half_time_window = time_window // 2

    min_t = t_cur - half_time_window
    max_t = t_cur + half_time_window

    # Using bisect to find the closest index to t_cur
    pos_cur_t = bisect.bisect_left(ts, t_cur)
    if pos_cur_t == 0:
        idx_cur = pos_cur_t
    elif pos_cur_t == len(ts):
        idx_cur = pos_cur_t - 1
    else:
        # Determine the closest index by comparing distances to the adjacent values
        idx_cur = pos_cur_t if abs(ts[pos_cur_t] - min_t) < abs(ts[pos_cur_t - 1] - min_t) else pos_cur_t - 1

    # Using bisect to find the closest index to min_t
    pos_min_t = bisect.bisect_left(ts, min_t)
    if pos_min_t == 0:
        idx_start = pos_min_t
    elif pos_min_t == len(ts):
        idx_start = pos_min_t - 1
    else:
        # Determine the closest index by comparing distances to the adjacent values
        idx_start = pos_min_t if abs(ts[pos_min_t] - min_t) < abs(ts[pos_min_t - 1] - min_t) else pos_min_t - 1
    
    # Using bisect to find the closest index to max_t
    pos_max_t = bisect.bisect_left(ts, max_t)
    if pos_max_t == 0:
        idx_end = pos_max_t
    elif pos_max_t == len(ts):
        idx_end = pos_max_t - 1
    else:
        # Determine the closest index by comparing distances to the adjacent values
        idx_end = pos_max_t if abs(ts[pos_max_t] - max_t) < abs(ts[pos_max_t - 1] - max_t) else pos_max_t - 1


    return idx_start, idx_cur, idx_end


def load_map(map_file, device):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = torch.tensor(downpcd.points, dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    voxelized = voxelized.to(device)
    return voxelized

def pc_visualize(pointcloud):
    point_cloud = o3.open3d.geometry.PointCloud()
    point_cloud.points = o3.open3d.utility.Vector3dVector(pointcloud[:,0:3].reshape(-1,3))
    o3.open3d.visualization.draw_geometries([point_cloud],width=800,height=600)
    return point_cloud


def depth_generation(local_map, image_size, cam_params, occu_thre, occu_kernel, device, only_uv=False):
    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, _, refl, indexes = cam_model.project_withindex_pytorch(local_map, image_size)
    uv = uv.t().int().contiguous()
    if only_uv:
        return uv, indexes
    depth_img = torch.zeros(image_size[:2], device=device, dtype=torch.float)
    depth_img += 1000.
    idx_img = (-1) * torch.ones(image_size[:2], device=device, dtype=torch.float)
    indexes = indexes.float()

    depth_img, idx_img = visibility.depth_image(uv, depth, indexes,
                                                depth_img, idx_img,
                                                uv.shape[0], image_size[1], image_size[0])
    depth_img[depth_img == 1000.] = 0.

    deoccl_index_img = (-1) * torch.ones(image_size[:2], device=device, dtype=torch.float)
    projected_points = torch.zeros_like(depth_img, device=device)
    projected_points, _ = visibility.visibility2(depth_img, cam_params,
                                                 idx_img,
                                                 projected_points,
                                                 deoccl_index_img,
                                                 depth_img.shape[1],
                                                 depth_img.shape[0],
                                                 occu_thre,
                                                 int(occu_kernel))
    projected_points /= 10.
    projected_points = projected_points.unsqueeze(0)

    return projected_points


def apply_contrast_stretching(image, low, high):
    # Ensure the low and high values are within the valid range [0, 255]
    # Create a copy of the image to avoid modifying the original
    stretched_image = image.copy()

    # Apply the contrast stretching to each pixel
    stretched_image = np.where(stretched_image < low, 0, stretched_image)
    stretched_image = np.where((low <= stretched_image) & (stretched_image <= high),
                              (255 / (high - low)) * (stretched_image - low), stretched_image)
    stretched_image = np.where(stretched_image > high, 255, stretched_image)

    return stretched_image

def calculate_L(j, i_T):
    return 1 if j >= i_T else 0
def calculate_D(k):
    return 1 if k >= 1 else 0
def calculate_DLBP(center_pixel, neighbors):
    i_max = np.max(neighbors)
    i_aver = np.mean(neighbors)
    i_T = i_max - i_aver
    LBP_value = sum(calculate_L(neighbor - center_pixel, i_T) for neighbor in neighbors)
    DLBP_value = calculate_D(LBP_value)
    return DLBP_value
def calculate_R(center_pixel, neighbors, threshold=1.5):
    i_aver = np.mean(neighbors)
    return calculate_DLBP(center_pixel, neighbors) if abs(center_pixel - i_aver) >= threshold else 0

def remove_isolated_edges(depth_edge_map, mask_size):
    h, w = depth_edge_map.shape
    half_mask = mask_size // 2
    
    # Create a mask for filtering
    mask = np.ones((mask_size, mask_size), dtype=np.uint8)
    
    # Iterate through the image, excluding the border
    for y in range(half_mask, h - half_mask):
        for x in range(half_mask, w - half_mask):
            # Extract the neighborhood of the current pixel
            neighborhood = depth_edge_map[y - half_mask:y + half_mask + 1, x - half_mask:x + half_mask + 1]
            # Check if any pixel in the neighborhood exceeds the threshold
            if (np.all(neighborhood[0, :] != 1) and np.all(neighborhood[-1, :] != 1) and np.all(neighborhood[:, 0] != 1) and np.all(neighborhood[:, -1] != 1)):
              # Set all elements to 0
              depth_edge_map[y - half_mask:y + half_mask + 1, x - half_mask:x + half_mask + 1] = np.zeros_like(neighborhood)
            elif (np.all(neighborhood[1, :] != 1) and np.all(neighborhood[-2, :] != 1) and np.all(neighborhood[:, 1] != 1) and np.all(neighborhood[:, -2] != 1)) :
              depth_edge_map[y,x]=0
    return depth_edge_map



def calculate_DLBP_torch(patch):
    """
        patch: tensor HxWx3x3
    """
    center_pixel = patch[:, :, 1, 1].clone()
    patch_clone = patch.clone()
    patch_clone[:, :, 1, 1] = -999.
    i_max_temp, _ = torch.max(patch_clone, dim=2)
    i_max, _ = torch.max(i_max_temp, dim=2)
    i_aver = (torch.sum(patch.clone(), dim=[2,3]) - patch[:, :, 1, 1]) / 8.
    i_T = i_max - i_aver
    LBP_value = patch.clone() - center_pixel.unsqueeze(-1).unsqueeze(-1)
    Mask = LBP_value >= i_T.unsqueeze(-1).unsqueeze(-1)
    LBP_value[Mask] = 1
    LBP_value[~Mask] = 0
    LBP_value[:, :, 1, 1] = 1
    LBP_value = torch.sum(LBP_value, dim=[2,3])
    DLBP_value = torch.where(LBP_value>=1, 1, 0)

    return DLBP_value

def calculate_R_torch(patch, threshold=1.5):
    """
        patch: tensor HxWx3x3
    """
    patch_aver = (torch.sum(patch.clone(), dim=[2,3]) - patch[:, :, 1, 1]) / 8.
    # return calculate_DLBP_torch(patch) if abs(center_pixel - i_aver) >= threshold else 0
    mask = (torch.abs(patch[:, :, 1, 1]) - patch_aver) >= threshold

    patch = calculate_DLBP_torch(patch)

    patch[~mask] = 0

    patch[0, :] = 0
    patch[-1, :] = 0
    patch[:, 0] = 0
    patch[:, -1] = 0

    return patch

def remove_isolated_edges_torch(depth_edge_map, mask_size):
    H, W = depth_edge_map.shape
    half_mask = mask_size // 2
    coords = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords = torch.stack(coords[::-1], dim=0).float().to(depth_edge_map.device)
    coords = coords[None]
    dx = torch.linspace(-half_mask, half_mask, 2 * half_mask + 1)
    dy = torch.linspace(-half_mask, half_mask, 2 * half_mask + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(depth_edge_map.device)
    centroid_lvl = coords.permute(0, 2, 3, 1).reshape(1*H*W, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * half_mask + 1, 2 * half_mask + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl   # BHW x 3 x 3 x 2
    patch = bilinear_sampler((depth_edge_map.unsqueeze(0).unsqueeze(0)).float(), coords_lvl.reshape(1, H*W*(2 * half_mask + 1), 2 * half_mask + 1, 2))
    patch = patch.reshape(H, W, 2 * half_mask + 1, 2 * half_mask + 1)    # H x W x 5 x 5

    # if (np.all(neighborhood[0, :] != 1) and np.all(neighborhood[-1, :] != 1) and np.all(neighborhood[:, 0] != 1) and np.all(neighborhood[:, -1] != 1)):
    #     # Set all elements to 0
    #     depth_edge_map[y - half_mask:y + half_mask + 1, x - half_mask:x + half_mask + 1] = np.zeros_like(neighborhood)
    # elif (np.all(neighborhood[1, :] != 1) and np.all(neighborhood[-2, :] != 1) and np.all(neighborhood[:, 1] != 1) and np.all(neighborhood[:, -2] != 1)) :
    #     depth_edge_map[y,x]=0

    mask = torch.sum(patch, dim=[2, 3]) - torch.sum(patch[:, :, 1:-1, 1:-1], dim=[2, 3]) == 0
    mask_2 = (torch.sum(patch[:, :, 1:-1, 1:-1], dim=[2, 3]) - patch[:, :, 2, 2] == 0) * (~mask)

    index_u, index_v = torch.where(mask>0)
    index = torch.cat((index_u.unsqueeze(-1), index_v.unsqueeze(-1)), dim=1)
    result_index = index.clone()
    for i in range(-half_mask, half_mask):
        for j in range(-half_mask, half_mask):
            if i == 0 and j == 0:
                continue
            else:
                index_clone = index.clone()
                index_clone[:, 0] -= i
                index_clone[:, 1] -= j
                result_index = torch.cat((result_index, index_clone), dim=0)
    valid = (result_index[:, 0] >= 0) * (result_index[:, 1] >= 0) * (result_index[:, 0] < H) * (result_index[:, 1] < W)
    result_index = result_index[valid]

    depth_edge_map[result_index[:, 0], result_index[:, 1]] = 0

    depth_edge_map[mask_2] = 0

    return depth_edge_map

def enhanced_depth_line_extract(image):
    # Define the low and high values for contrast stretching
    low_value = np.percentile(image, 1)  # 1st percentile
    high_value = np.percentile(image, 99)  # 99th percentile
    # Apply contrast stretching to the image using the provided function
    stretched_image = apply_contrast_stretching(image, low_value, high_value)
    stretched_image = np.uint8(stretched_image)

    # image = stretched_image.copy()
    # output_image = np.zeros_like(image)
    # neighborhood_size = 3
    # # Iterate through the image pixels, applying DLBP and edge detection
    # for y in range(neighborhood_size, image.shape[0] - neighborhood_size):
    #     for x in range(neighborhood_size, image.shape[1] - neighborhood_size):
    #         center_pixel = image[y, x]
    #         neighbors = [image[y-1, x-1], image[y-1, x], image[y-1, x+1],
    #                     image[y, x-1], image[y, x+1],
    #                     image[y+1, x-1], image[y+1, x], image[y+1, x+1]]
            
    #         R_value = calculate_R(center_pixel, neighbors)
    #         output_image[y, x] = R_value

    # depth_edge_map = output_image.copy()
    # result = remove_isolated_edges(depth_edge_map, mask_size=5)

    image = torch.tensor(stretched_image.copy())
    H, W = image.shape
    neighborhood_size = 1
    coords = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords = torch.stack(coords[::-1], dim=0).float().to(image.device)
    coords = coords[None]
    dx = torch.linspace(-neighborhood_size, neighborhood_size, 2 * neighborhood_size + 1)
    dy = torch.linspace(-neighborhood_size, neighborhood_size, 2 * neighborhood_size + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(image.device)
    centroid_lvl = coords.permute(0, 2, 3, 1).reshape(1*H*W, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * neighborhood_size + 1, 2 * neighborhood_size + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl   # BHW x 3 x 3 x 2
    patch = bilinear_sampler((image.unsqueeze(0).unsqueeze(0)).float(), coords_lvl.reshape(1, H*W*(2 * neighborhood_size + 1), 2 * neighborhood_size + 1, 2))
    patch = patch.reshape(H, W, 2 * neighborhood_size + 1, 2 * neighborhood_size + 1)    # H x W x 3 x 3
    output_image = calculate_R_torch(patch)

    depth_edge_map = output_image.clone()
    result = remove_isolated_edges_torch(depth_edge_map, mask_size=5)

    return stretched_image, output_image, result


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def remove_noise(input_depth_map, radius=3, threshold=2):
    depth_map = input_depth_map.clone()
    B, H, W = depth_map.shape
    coords = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords = torch.stack(coords[::-1], dim=0).float().to(depth_map.device)
    coords = coords[None].repeat(B, 1, 1, 1)

    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(depth_map.device)


    centroid_lvl = coords.permute(0, 2, 3, 1).reshape(B*H*W, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * radius + 1, 2 * radius + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl                               # BHW x 11 x 11 x 2
    
    # patch = bilinear_sampler((depth_map.unsqueeze(1)>0).float(), coords_lvl.reshape(B, H*W*(2 * radius + 1), 2 * radius + 1, 2))
    patch = bilinear_sampler((depth_map.unsqueeze(1)>0).float(), coords_lvl.reshape(B, H*W*(2 * radius + 1), 2 * radius + 1, 2))
    patch = patch.reshape(B, H, W, 2 * radius + 1, 2 * radius + 1)

    patch = torch.sum(patch>0, dim=[3, 4])
    alone_point = patch < threshold

    depth_map[alone_point] = 0

    return depth_map