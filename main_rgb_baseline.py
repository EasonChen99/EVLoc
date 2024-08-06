import os
import sys
import time 

import cv2
import numpy as np
import argparse
import torch

import visibility

from core.datasets_m3ed_rgb import DatasetM3ED
from core.backbone import Backbone
from core.utils import (count_parameters, merge_inputs_rgb, fetch_optimizer, Logger)
from core.utils_point import overlay_imgs
from core.data_preprocess import Data_preprocess
from core.flow_viz import flow_to_image
from core.flow2pose import Flow2Pose, err_Pose
from core.losses import sequence_loss, uncertainty_loss

occlusion_kernel = 5
occlusion_threshold = 3
seed = 1234

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass
    

def _init_fn(worker_id, seed):
    seed = seed
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)


def train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device):
    global occlusion_threshold, occlusion_kernel
    model.train()
    for i_batch, sample in enumerate(TrainImgLoader):
        rgb = sample['rgb']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        rgb_input, lidar_input, flow_gt = data_generate.push(rgb, pc, T_err, R_err, device)

        vis_rgb_input = overlay_imgs(rgb_input[0, :, :, :], 0*lidar_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/{i_batch:05d}_rgb.png", vis_rgb_input)
        vis_lidar_input = overlay_imgs(rgb_input[0, :, :, :]*0, lidar_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/{i_batch:05d}_projection.png", vis_lidar_input)
        # flow_image = flow_to_image(flow_gt.permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        # flow_image[flow_image == 255] = 0
        # cv2.imwrite(f'./visualization/{i_batch:05d}_flow.png', flow_image)

        optimizer.zero_grad()
        if not args.use_uncertainty:
            flow_preds = model(lidar_input, rgb_input, iters=args.iters)
            loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
        else:
            flow_preds, uncertainty_maps = model(lidar_input, rgb_input, iters=args.iters) 
            loss, metrics = uncertainty_loss(flow_preds, uncertainty_maps, flow_gt, args.gamma, MAX_FLOW=400)
        

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        logger.push(metrics)

def test(args, TestImgLoader, model, device, cal_pose=False):
    global occlusion_threshold, occlusion_kernel
    model.eval()
    out_list, epe_list = [], []
    Time = 0.
    outliers, err_r_list, err_t_list = [], [], []
    for i_batch, sample in enumerate(TestImgLoader):
        rgb = sample['rgb']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        rgb_input, lidar_input, flow_gt = data_generate.push(rgb, pc, T_err, R_err, device, split='test')

        end = time.time()
        if not args.use_uncertainty:
            _, flow_up = model(lidar_input, rgb_input, iters=24, test_mode=True)
        else:
            _, flow_up, uncertainty_map = model(lidar_input, rgb_input, iters=24, test_mode=True)

        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)

        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        if not np.isnan(epe[val].mean().item()):
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())

        if cal_pose:
            R_preds, T_preds = [], []
            for i in range(5):
                R_pred, T_pred, flag = Flow2Pose(flow_up, lidar_input, calib)
                if flag:
                    break
                R_preds.append(R_pred.cpu().numpy())
                T_preds.append(T_pred.cpu().numpy())
            R_pred = torch.tensor(np.mean(R_preds, axis=0))
            T_pred = torch.tensor(np.mean(T_preds, axis=0))
            # R_pred, T_pred, _ = Flow2Pose(flow_up, lidar_input, calib)

            Time += time.time() - end
            if flag:
                outliers.append(i_batch)
            else:
                err_r, err_t = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
                err_r_list.append(err_r.item())
                err_t_list.append(err_t.item())
            print(f"{i_batch:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.median(err_t_list):.5f} "
                  f"{np.median(err_r_list):.5f} {len(outliers)} {Time / (i_batch+1):.5f}")
        
        if args.render:
            if not os.path.exists(f"./visualization/test"):
                os.makedirs(f"./visualization/test")

            # vis_rgb = overlay_imgs(rgb_input[0, :, :, :], lidar_input[0, 0, :, :])
            # cv2.imwrite(f'./visualization/test/{i_batch:05d}.png', vis_rgb)

            vis_rgb = overlay_imgs(rgb_input[0, :, :, :], 0 * lidar_input[0, 0, :, :])

            flow_image = flow_to_image(flow_up.permute(0, 2, 3, 1).cpu().detach().numpy()[0])

            output = torch.zeros(flow_up.shape).to(device)
            pred_depth_img = torch.zeros(lidar_input.shape).to(device)
            pred_depth_img += 1000.
            output = visibility.image_warp_index(lidar_input.to(device),
                                                 flow_up.int().to(device), pred_depth_img,
                                                 output, lidar_input.shape[3], lidar_input.shape[2])
            pred_depth_img[pred_depth_img == 1000.] = 0.

            original_overlay = overlay_imgs(rgb_input[0, :, :, :]*0, lidar_input[0, 0, :, :])
            warp_overlay = overlay_imgs(rgb_input[0, :, :, :]*0, pred_depth_img[0, 0, :, :])
            valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
            errors = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt() * valid_gt
            errors = 255 - (errors - torch.min(errors)) / (torch.max(errors) - torch.min(errors)) * 255
            errors = errors * valid_gt
            errors = torch.cat((errors, torch.zeros([2, errors.shape[1], errors.shape[2]], device=device)), dim=0).permute(1, 2, 0).cpu().detach().numpy()[:, :, [2, 1, 0]]

            cat_ori = np.hstack((vis_rgb, original_overlay))
            cat_pre = np.hstack((flow_image, warp_overlay))
            # cat_pre = np.hstack((flow_image, errors))
            cat = np.vstack((cat_ori, cat_pre))

            # Define text properties
            text = f"R={err_r.item():.5f} T={err_t.item():.5f}"
            org = (cat.shape[1]-450, cat.shape[0]-30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            # Add text to the image
            cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.imwrite(f'./visualization/test/{i_batch:05d}.png', cat)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    if not cal_pose:
        return epe, f1
    else:
        return err_t_list, err_r_list, outliers, Time, epe, f1   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        metavar='DIR',
                        default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED',
                        help='path to dataset')
    parser.add_argument('--suffix',
                        type=str,
                        default='100000',
                        help='suffix of the event frame folder')
    parser.add_argument('--use_uncertainty',
                        action='store_true',
                        help='estimate uncertainty map')
    parser.add_argument('--test_sequence',
                        type=str, 
                        default='falcon_indoor_flight_3')
    parser.add_argument('--load_checkpoints',
                        help="restore checkpoint")
    parser.add_argument('--epochs', 
                        default=100, 
                        type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--starting_epoch', 
                        default=0, 
                        type=int, 
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', 
                        default=2, 
                        type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', 
                        '--learning_rate', 
                        default=4e-5, 
                        type=float,
                        metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--wdecay', 
                        type=float, 
                        default=.00005)
    parser.add_argument('--epsilon', 
                        type=float, 
                        default=1e-8)
    parser.add_argument('--clip', 
                        type=float, 
                        default=1.0)
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.8, 
                        help='exponential weighting')
    parser.add_argument('--iters', 
                        type=int, 
                        default=12)
    parser.add_argument('--gpus', 
                        type=int, 
                        nargs='+', 
                        default=[0])
    parser.add_argument('--max_r', 
                        type=float, 
                        default=5.)
    parser.add_argument('--max_t', 
                        type=float, 
                        default=0.5)
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=3)
    parser.add_argument('--mixed_precision', 
                        action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--evaluate_interval', 
                        default=1, 
                        type=int, 
                        metavar='N',
                        help='Evaluate every \'evaluate interval\' epochs ')
    parser.add_argument('-e', 
                        '--evaluate', 
                        dest='evaluate', 
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--render', 
                        action='store_true')
    args = parser.parse_args()    


    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])

    batch_size = args.batch_size

    model = torch.nn.DataParallel(Backbone(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    if args.load_checkpoints is not None:
        model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)

    def init_fn(x):
        return _init_fn(x, seed)

    dataset_test = DatasetM3ED(args.data_path,
                               args.suffix, 
                               max_r=args.max_r, 
                               max_t=args.max_t,
                               split='test', 
                               test_sequence=args.test_sequence)
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs_rgb,
                                                drop_last=False,
                                                pin_memory=True)
    if args.evaluate:
        with torch.no_grad():
            err_t_list, err_r_list, outliers, Time, epe, f1 = test(args, TestImgLoader, model, device, cal_pose=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"Outliers number {len(outliers)}/{len(TestImgLoader)}  Mean {Time / len(TestImgLoader):.5f} per frame")
            print(f"epe {epe:.5f}")
        sys.exit()

    dataset_train = DatasetM3ED(args.data_path,
                                args.suffix, 
                                max_r=args.max_r, 
                                max_t=args.max_t,
                                split='train',
                                test_sequence=args.test_sequence)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=args.num_workers,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs_rgb,
                                                 drop_last=False,
                                                 pin_memory=True)
    print("Train length: ", len(TrainImgLoader))
    print("Test length: ", len(TestImgLoader))

    optimizer, scheduler = fetch_optimizer(args, len(TrainImgLoader), model)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, SUM_FREQ=100)

    datetime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    if not os.path.exists(f'./checkpoints/{datetime}'):
        os.mkdir(f'./checkpoints/{datetime}')

    starting_epoch = args.starting_epoch
    min_val_err = 9999.
    for epoch in range(starting_epoch, args.epochs):
        # train
        train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device)

        if epoch % args.evaluate_interval == 0:
            epe, f1 = test(args, TestImgLoader, model, device)
            print("Validation M3ED: %f, %f" % (epe, f1))

            results = {'m3ed-epe': epe, 'm3ed-f1': f1}
            logger.write_dict(results)

            torch.save(model.state_dict(), f"./checkpoints/{datetime}/checkpoint.pth")

            if epe < min_val_err:
                min_val_err = epe
                torch.save(model.state_dict(), f'./checkpoints/{datetime}/best_model.pth')