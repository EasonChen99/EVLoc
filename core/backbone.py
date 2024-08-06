import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.update import BasicUpdateBlock
from core.extractor import BasicEncoder_Event, BasicEncoder_LiDAR, BasicEncoder_RGB, SparseEncoder_Edge
from core.corr import CorrBlock, AlternateCorrBlock, OffsetCorrBlock
from core.utils import coords_grid, upflow8
from core.utils_point import invert_pose

import mathutils

import spconv.pytorch as spconv

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class Attention(nn.Module):
    def __init__(self, emb_dim):
        super(Attention, self).__init__()
        self.emb_dim = emb_dim
 
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)
 
        self.fc = nn.Linear(emb_dim, emb_dim)
 
    def forward(self, x, context, pad_mask=None):
        Q = self.Wq(context)
        K = self.Wk(x)
        V = self.Wv(x)
 
        att_weights = torch.bmm(Q, K.transpose(1, 2))
        att_weights = att_weights / math.sqrt(self.emb_dim)
 
        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)
 
        att_weights = F.softmax(att_weights, dim=-1)
        output = torch.bmm(att_weights, V)
        output = self.fc(output)
 
        return output, att_weights


class OffsetHead_RT(nn.Module):
    def __init__(self, input_dim):
        super(OffsetHead_RT, self).__init__()
        self.conva1 = nn.Conv2d(input_dim, 512, 3, stride=2, padding=1)
        self.conva2 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.convb1 = nn.Conv2d(input_dim, 512, 3, stride=2, padding=1)
        self.convb2 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)

        self.norm1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.leakyRELU = nn.LeakyReLU(0.1)

        fc_size = 19 * 30 * 19 * 30
        self.fc1 = nn.Linear(fc_size, 512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fmap1, fmap2):
        B, _, _, _ = fmap1.shape

        fmap1 = self.conva2(self.relu(self.norm1(self.conva1(fmap1))))
        fmap2 = self.convb2(self.relu(self.norm1(self.convb1(fmap2))))

        corr = CorrBlock.corr(fmap1, fmap2)
        corr = corr.view(B, -1)

        x = self.dropout(corr)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)

        transl = 0.1 * torch.tanh(transl)
        angle_range = 1/180 * torch.pi  # For [-π, π] range
        rot = angle_range * torch.tanh(rot)

        return transl, rot


class OffsetHead_UV(nn.Module):
    def __init__(self, input_dim):
        super(OffsetHead_UV, self).__init__()
        # self.conv1 = nn.Conv2d(input_dim, 512, 3, padding=1)
        # self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 2, 3, padding=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

    def forward(self, fmap1, fmap2):
        fmap_cat = torch.cat((fmap1, fmap2), dim=1)

        # offset_map = self.conv3(self.leakyRELU(self.conv2(self.leakyRELU(self.conv1(fmap_cat)))))
        offset_map = self.conv3(self.leakyRELU(self.conv2(fmap_cat)))

        # offset_map = torch.tanh(offset_map)

        return offset_map


class UncertaintyHead(nn.Module):
    def __init__(self, input_dim):
        super(UncertaintyHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(64)

    def forward(self, fmap1, fmap2):
        dist_fmap = fmap1 - fmap2

        uncertainty_map = self.conv2(self.relu(self.conv1(dist_fmap)))

        return uncertainty_map


class Backbone_Event(nn.Module):
    def __init__(self, args):
        super(Backbone_Event, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # print(torch.min(image1), torch.max(image1))
        # print(torch.min(image2), torch.max(image2))

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        # # visualize feature map
        # upsample = torch.nn.Upsample(size=(fmap1.shape[2]*8, fmap1.shape[3]*8), mode='bilinear')
        # B, C, H, W = fmap1.shape
        # # fmap1_vis = upsample(fmap1).squeeze(0)
        # # fmap1_vis = torch.sum(fmap1_vis, 0) / fmap1_vis.shape[0]
        # # fmap1_vis = fmap1_vis / torch.max(fmap1_vis) * 255
        # # fmap2_vis = upsample(fmap2).squeeze(0)
        # # fmap2_vis = torch.sum(fmap2_vis, 0) / fmap2_vis.shape[0]
        # # fmap2_vis = fmap2_vis / torch.max(fmap2_vis) * 255
        # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # cos_map = cos(fmap1.squeeze(0).view(C, -1), fmap2.squeeze(0).view(C, -1))
        # cos_map = cos_map.reshape(H, W)
        # cos_map = upsample(cos_map.unsqueeze(0).unsqueeze(0))
        # cos_map = (cos_map / torch.max(cos_map)) * 255
        # import cv2
        # import datetime
        # cv2.imwrite(f"/home/eason/WorkSpace/EventbasedVisualLocalization/E2D_Loc/visualization/test/feature_map/feature_map_{datetime.datetime.now()}.png", cos_map[0, 0, ...].cpu().numpy())

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions


class Backbone_RGB(nn.Module):
    def __init__(self, args):
        super(Backbone_RGB, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_RGB(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions


class Backbone_Event_Offset_RT(nn.Module):
    def __init__(self, args):
        super(Backbone_Event_Offset_RT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.offset_head = OffsetHead_RT(input_dim=256)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0, 2, 3, 1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, encode_only=False):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # print(torch.min(image1), torch.max(image1))
        # print(torch.min(image2), torch.max(image2))

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            if not encode_only:
                fmap2 = self.fnet_event(image2)

        if encode_only:
            return fmap1

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        offsets_R, offsets_T = [], []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            warp_fmap1 = self.warp(fmap1, coords1-coords0)
            offset_T, offset_R = self.offset_head(warp_fmap1, fmap2)

            offset_T_batch, offset_R_batch = [], []
            for i in range(offset_R.shape[0]):
                R = mathutils.Euler((offset_R[i, 0], offset_R[i, 1], offset_R[i, 2]), 'XYZ')
                T = mathutils.Vector((offset_T[i, 0], offset_T[i, 1], offset_T[i, 2]))
                R, T = invert_pose(R, T)
                R, T = torch.tensor(R), torch.tensor(T)
                offset_T_batch.append(T)
                offset_R_batch.append(R)
            offset_T = torch.stack(offset_T_batch).to(fmap1.device)
            offset_R = torch.stack(offset_R_batch).to(fmap1.device)

            offsets_T.append(offset_T)
            offsets_R.append(offset_R)

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, offset_R, offset_T
            
        return flow_predictions, offsets_R, offsets_T, fmap2


class Backbone_Event_Offset_RT_Atten(nn.Module):
    def __init__(self, args):
        super(Backbone_Event_Offset_RT_Atten, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.offset_head = OffsetHead_RT(input_dim=256)

        self.fnet_edge = SparseEncoder_Edge(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.atten = Attention(emb_dim=256)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0, 2, 3, 1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, edge, iters=12, flow_init=None, test_mode=False, encode_only=False):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0
        edge = 2 * edge - 1.0

        # print(torch.min(image1), torch.max(image1))
        # print(torch.min(image2), torch.max(image2))
        # print(torch.min(edge), torch.max(edge))

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        edge = edge.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            if not encode_only:
                fmap2 = self.fnet_event(image2)

        if encode_only:
            return fmap1

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # Attention
        edge_sparse = spconv.SparseConvTensor.from_dense(edge.permute(0, 2, 3, 1))
        fmap_edge = self.fnet_edge(edge_sparse)
        fmap_edge = fmap_edge.dense()

        B, C, H, W = fmap1.shape
        fmap1, _ = self.atten(fmap1.reshape(B, C, H*W).permute(0, 2, 1), fmap_edge.reshape(B, C, H*W).permute(0, 2, 1))
        fmap1 = fmap1.permute(0, 2, 1).reshape(B, C, H, W)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        offsets_R, offsets_T = [], []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            warp_fmap1 = self.warp(fmap1, coords1-coords0)
            offset_T, offset_R = self.offset_head(warp_fmap1, fmap2)

            offset_T_batch, offset_R_batch = [], []
            for i in range(offset_R.shape[0]):
                R = mathutils.Euler((offset_R[i, 0], offset_R[i, 1], offset_R[i, 2]), 'XYZ')
                T = mathutils.Vector((offset_T[i, 0], offset_T[i, 1], offset_T[i, 2]))
                R, T = invert_pose(R, T)
                R, T = torch.tensor(R), torch.tensor(T)
                offset_T_batch.append(T)
                offset_R_batch.append(R)
            offset_T = torch.stack(offset_T_batch).to(fmap1.device)
            offset_R = torch.stack(offset_R_batch).to(fmap1.device)

            offsets_T.append(offset_T)
            offsets_R.append(offset_R)

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, offset_R, offset_T
            
        return flow_predictions, offsets_R, offsets_T, fmap2


class Backbone_Event_Offset_UV(nn.Module):
    def __init__(self, args):
        super(Backbone_Event_Offset_UV, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.offset_head = OffsetHead_UV(input_dim=512)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = OffsetCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        offset_flows = []
        offset_flow = None
        uncertainty_maps = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr, uncertainty = corr_fn(coords1, offset_flow)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            warp_fmap1 = self.warp(fmap1, coords1-coords0)
            offset_flow = self.offset_head(warp_fmap1, fmap2)
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
                offset_flow_up = upflow8(offset_flow)
                uncertainty_up = upflow8(uncertainty)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                offset_flow_up = self.upsample_flow(offset_flow, up_mask)
                uncertainty_up = self.upsample_flow(uncertainty, up_mask)

            flow_predictions.append(flow_up)
            offset_flows.append(offset_flow_up)
            uncertainty_maps.append(uncertainty_up)

        if test_mode:
            return coords1 - coords0, flow_up, offset_flow_up, uncertainty_up
            
        return flow_predictions, offset_flows, uncertainty_maps


class Backbone_Event_Offset_RT_UV(nn.Module):
    def __init__(self, args):
        super(Backbone_Event_Offset_RT_UV, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.offset_head_rt = OffsetHead_RT(input_dim=256)

        self.offset_head_uv = OffsetHead_UV(input_dim=512)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = OffsetCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        offsets_R, offsets_T = [], []
        offset_flow = None
        uncertainty_maps = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr, uncertainty = corr_fn(coords1, offset_flow)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            warp_fmap1 = self.warp(fmap1, coords1-coords0)
            offset_T, offset_R = self.offset_head_rt(warp_fmap1, fmap2)
            offset_flow = self.offset_head_uv(warp_fmap1, fmap2)

            offset_T_batch, offset_R_batch = [], []
            for i in range(offset_R.shape[0]):
                R = mathutils.Euler((offset_R[i, 0], offset_R[i, 1], offset_R[i, 2]), 'XYZ')
                T = mathutils.Vector((offset_T[i, 0], offset_T[i, 1], offset_T[i, 2]))
                R, T = invert_pose(R, T)
                R, T = torch.tensor(R), torch.tensor(T)
                offset_T_batch.append(T)
                offset_R_batch.append(R)
            offset_T = torch.stack(offset_T_batch).to(fmap1.device)
            offset_R = torch.stack(offset_R_batch).to(fmap1.device)

            offsets_T.append(offset_T)
            offsets_R.append(offset_R)

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
                uncertainty_up = upflow8(uncertainty)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                uncertainty_up = self.upsample_flow(uncertainty, up_mask)

            flow_predictions.append(flow_up)
            uncertainty_maps.append(uncertainty_up)

        if test_mode:
            return coords1 - coords0, flow_up, offset_R, offset_T, uncertainty_up, offset_flow
            
        return flow_predictions, offsets_R, offsets_T, uncertainty_maps