import torch.optim as optim
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torch.utils.data.dataloader import default_collate

class Logger:
    def __init__(self, model, scheduler, SUM_FREQ=100):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.SUM_FREQ = SUM_FREQ

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, nums, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.epochs * nums + 100,
    #                                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs * nums + 100, gamma=0.1)
    return optimizer, scheduler


def merge_inputs(queries):
    point_clouds = []
    imgs = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'event_frame'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['event_frame'])
    returns['point_cloud'] = point_clouds
    returns['event_frame'] = imgs
    return returns

def merge_inputs_rgb(queries):
    point_clouds = []
    imgs = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    return returns


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


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

    return stretched_image, output_image

    # depth_edge_map = output_image.clone()
    # result = remove_isolated_edges_torch(depth_edge_map, mask_size=5)

    # return stretched_image, output_image, result