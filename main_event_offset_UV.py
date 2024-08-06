import os
import sys
import time 

import cv2
import numpy as np
import argparse
import torch

import visibility

from core.datasets_m3ed import DatasetM3ED
from core.backbone import Backbone_Event_Offset_UV
from core.utils import (count_parameters, merge_inputs, fetch_optimizer, Logger)
from core.utils_point import (overlay_imgs, to_rotation_matrix, quaternion_from_matrix)
from core.data_preprocess import Data_preprocess
from core.flow_viz import flow_to_image
from core.flow2pose import Flow2Pose, err_Pose, warp
from core.losses import sequence_loss, sequence_loss_single, uncertainty_loss_2

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


def train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch):
    global occlusion_threshold, occlusion_kernel
    model.train()
    for i_batch, sample in enumerate(TrainImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device)

        vis_event_time_image = event_input[0,...].permute(1, 2, 0).cpu().numpy()
        vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        cv2.imwrite(f"./visualization/{i_batch:05d}_event.png", (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8))
        vis_lidar_input = overlay_imgs(event_input[0, :, :, :]*0, lidar_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/{i_batch:05d}_projection.png", (vis_lidar_input / np.max(vis_lidar_input) * 255).astype(np.uint8))
        # flow_image = flow_to_image(flow_gt.permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        # flow_image[flow_image == 255] = 0
        # # cv2.imwrite(f'./visualization/{i_batch:05d}_flow.png', flow_image)

        optimizer.zero_grad()
        flow_preds, offset_flows, uncertainty_maps = model(lidar_input, event_input, iters=args.iters)
        # loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
        loss, metrics = uncertainty_loss_2(flow_preds, uncertainty_maps, flow_gt, args.gamma, MAX_FLOW=400)

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
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, split='test')
        
        end = time.time()
        _, flow_up, offset_flow, uncertainty_map = model(lidar_input, event_input, iters=24, test_mode=True)

        vis_uncertainty_map = (uncertainty_map[0, 0, ...].cpu().detach().numpy()*255).astype(np.uint8)
        cv2.imwrite(f'./visualization/test/uncertainty/{i_batch:05d}_uncertainty.png', vis_uncertainty_map)

        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)

        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        if cal_pose:
            R_pred, T_pred, inliers, flag = Flow2Pose(flow_up, lidar_input, calib)

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
            if not os.path.exists(f"./visualization/test/cat"):
                os.makedirs(f"./visualization/test/cat")
            if not os.path.exists(f"./visualization/test/event"):
                os.makedirs(f"./visualization/test/event")
            if not os.path.exists(f"./visualization/test/depth"):
                os.makedirs(f"./visualization/test/depth") 
            if not os.path.exists(f"./visualization/test/flow"):
                os.makedirs(f"./visualization/test/flow")     
            if not os.path.exists(f"./visualization/test/uncertainty"):
                os.makedirs(f"./visualization/test/uncertainty")   

            vis_uncertainty_map = (uncertainty_map[0, 0, ...].cpu().detach().numpy()*255).astype(np.uint8)
            cv2.imwrite(f'./visualization/test/uncertainty/{i_batch:05d}_uncertainty.png', vis_uncertainty_map)

            vis_event_time_image = event_input[0, ...].permute(1, 2, 0).cpu().numpy()
            vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
            vis_event_time_image = (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8)
            cv2.imwrite(f'./visualization/test/event/{i_batch:05d}_event_1.png', vis_event_time_image)
            flow_image = flow_to_image(flow_up.permute(0, 2, 3, 1).cpu().detach().numpy()[0])
            cv2.imwrite(f'./visualization/test/flow/{i_batch:05d}_flow.png', flow_image)

            original_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input[0, 0, :, :])
            cv2.imwrite(f'./visualization/test/depth/{i_batch:05d}_depth_1.png', original_overlay)
            cat_ori = np.hstack((vis_event_time_image, original_overlay))

            original_overlay = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :])
            cat_pre = np.hstack((original_overlay, flow_image))
            cat = np.vstack((cat_ori, cat_pre))

            # error smaller, pixel brighter
            errors = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt() * valid_gt
            errors_visp = 255 - (errors - torch.min(errors)) / (torch.max(errors) - torch.min(errors)) * 255
            errors_visp = errors_visp * valid_gt
            errors_visp = torch.cat((errors_visp, torch.zeros([2, errors_visp.shape[1], errors_visp.shape[2]], device=device)), dim=0).permute(1, 2, 0).cpu().detach().numpy()[:, :, [2, 1, 0]]

            # vis inliers distribution
            inliers_map = np.zeros([lidar_input.shape[2], lidar_input.shape[3], 3])
            inliers_map[inliers[:, 0], inliers[:, 1], 0] = 1
            inliers_rate = inliers_map[:, :, 0].sum() / (lidar_input[0, 0, :, :]>0).sum()
            # inliers_map = inliers_map * np.repeat(valid_gt.cpu().numpy()[0, ...][:, :, None], repeats=3, axis=2)
            cat_inliers = np.vstack((inliers_map*255, errors_visp))
            cat = np.hstack((cat, cat_inliers))

            dist_map = np.zeros([lidar_input.shape[2], lidar_input.shape[3], 3])
            dist_map[(errors[0].cpu().numpy() < 5) * (errors[0].cpu().numpy() > 0), 1] = 1
            inliers_epe_5_rate = inliers_map[:, :, 0] * dist_map[:, :, 1]
            inliers_epe_5_rate = inliers_epe_5_rate.sum() / inliers_map[:, :, 0].sum()

            if cal_pose and (not flag):
                # Define text properties
                text = f"idx={i_batch:03d} R={err_r.item():.3f} T={err_t.item():.3f}"
                org = (cat.shape[1]-850, cat.shape[0]-70)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 0, 255)
                thickness = 2
                # Add text to the image
                cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                org = (cat.shape[1]-850, cat.shape[0]-40)
                text = f"inliers={inliers_map[:, :, 0].sum():.0f} inliers_rate={inliers_rate:.3f} epe={epe[val].mean().item()}"
                cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                org = (cat.shape[1]-850, cat.shape[0]-10)
                text = f"epe_5={dist_map[:, :, 1].sum():.0f} inliers_epe_5={inliers_epe_5_rate * inliers_map[:, :, 0].sum():.0f} inliers_epe_5={inliers_epe_5_rate:.3f}"
                cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.imwrite(f'./visualization/test/cat/{i_batch:05d}.png', cat)

            output = torch.zeros(flow_up.shape).to(device)
            pred_depth_img = torch.zeros(lidar_input.shape).to(device)
            pred_depth_img += 1000.
            output = visibility.image_warp_index(lidar_input.to(device), flow_up.int(), pred_depth_img, output,
                                                lidar_input.shape[3], lidar_input.shape[2])
            pred_depth_img[pred_depth_img == 1000.] = 0.
            # pc_project_uv = output.cpu().permute(0, 2, 3, 1).numpy()
            original_overlay = overlay_imgs(event_input[0, :, :, :]*0, pred_depth_img[0, 0, :, :])
            cv2.imwrite(f'./visualization/test/depth/{i_batch:05d}_depth_2.png', original_overlay)    

            vis_event_time_image = warp(event_input, offset_flow)
            vis_event_time_image = vis_event_time_image[0].permute(1, 2, 0).cpu().numpy()
            vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
            vis_event_time_image = (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8)
            cv2.imwrite(f'./visualization/test/event/{i_batch:05d}_event_2.png', vis_event_time_image)        


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

    model = torch.nn.DataParallel(Backbone_Event_Offset_UV(args), device_ids=args.gpus)
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
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    if args.evaluate:
        with torch.no_grad():
            err_t_list, err_r_list, outliers, Time, epe, f1 = test(args, TestImgLoader, model, device, cal_pose=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"epe {epe:.5f}  Mean {Time / len(TestImgLoader):.5f} per frame")
            print(f"Outliers number {len(outliers)}/{len(TestImgLoader)} {outliers}")
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
                                                 collate_fn=merge_inputs,
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
    if starting_epoch > 0:
        for i in range(starting_epoch * len(TrainImgLoader)):
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        logger.total_steps = starting_epoch * len(TrainImgLoader)

    min_val_err = 9999.
    for epoch in range(starting_epoch, args.epochs):
        # train
        train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch)

        torch.cuda.empty_cache()

        if epoch % args.evaluate_interval == 0:
            epe, f1 = test(args, TestImgLoader, model, device)
            print("Validation M3ED: %f, %f" % (epe, f1))

            results = {'m3ed-epe': epe, 'm3ed-f1': f1}
            logger.write_dict(results)

            torch.save(model.state_dict(), f"./checkpoints/{datetime}/checkpoint.pth")

            if epe < min_val_err:
                min_val_err = epe
                torch.save(model.state_dict(), f'./checkpoints/{datetime}/best_model.pth')
        
        torch.cuda.empty_cache()