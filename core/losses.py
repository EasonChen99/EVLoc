import torch
import torch.nn as nn
import numpy as np
import visibility

def sequence_loss(flow_preds, flow_gt, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (mask_sum + 1e-5)
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics

def sequence_loss_offset(flow_preds, flow_gt, offset_flow, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    flow_gt = flow_gt + offset_flow

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (mask_sum + 1e-5)
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics


def sequence_loss_single(flow_pred, flow_gt, Mask, mask_sum, uncertainty_map=None):
    """ Loss function defined over sequence of flow predictions """

    if uncertainty_map is None:
        Loss_reg = (flow_pred - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (mask_sum + 1e-5)
        flow_loss = Loss_reg.mean()
    else:
        flow_err = (flow_pred - flow_gt) ** 2
        flow_err = torch.exp(-1 * uncertainty_map) * flow_err
        flow_err += 2 * uncertainty_map
        flow_err = torch.sum(flow_err * Mask, dim=[1,2,3])
        flow_err = flow_err / (mask_sum + 1e-5)
        flow_loss = flow_err.mean()

    return flow_loss


def cal_cosine(tensor1, tensor2):
    """
        tensor1, tensor2: Bx2xHxW
    """
    # Compute the dot product over the channel dimension (dim=1)
    dot_product = (tensor1 * tensor2).sum(dim=1, keepdim=True)
    
    # Compute the magnitudes of the vectors
    magnitude1 = torch.sqrt((tensor1 ** 2).sum(dim=1, keepdim=True))
    magnitude2 = torch.sqrt((tensor2 ** 2).sum(dim=1, keepdim=True))
    
    # Calculate cosine similarity by dividing the dot product by the product of magnitudes
    cosine_similarity = dot_product / (magnitude1 * magnitude2 + 1e-5)

    return cosine_similarity, magnitude1

def uncertainty_loss(flow_preds, uncertainty_maps, flow_gt, gamma=0.8, MAX_FLOW=400, MIN_FLOW_ERR=1):
    '''
        The step size during each iteration is used to define the uncertainty of 
        the predicted optical flow in the current iteration.
            flow_preds:         LxBx2xHxW
            uncertainty_maps:   LxBx1xHxW
            flow_gt:            Bx2xHxW
    '''
    mag = torch.norm(flow_gt, dim=1)                                #BxHxW
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)  #BxHxW
    valid = mask & (mag < MAX_FLOW)                                 #BxHxW
    mask_sum = torch.sum(mask, dim=[1, 2])                          #B

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    pre_flow_error = 0.
    uncertainty_map_gts = []
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        
        flow_error = torch.norm(flow_preds[i] - flow_gt, dim=1, keepdim=True)    # Bx1xHxW

        delta_flow = flow_error - pre_flow_error

        # min-max normalization
        min_val = delta_flow.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = delta_flow.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        eps = 1e-5
        range_val = max_val - min_val
        range_val[range_val < eps] = eps
        uncertainty_map_gt = (delta_flow - min_val) / range_val         # Bx1xHxW
        uncertainty_map_gt = 1 - uncertainty_map_gt
        
        if i > 0:
            uncertainty_map_gt[(flow_error<MIN_FLOW_ERR) * (pre_flow_error<MIN_FLOW_ERR)] = 1.

        uncertainty_map_gts.append(uncertainty_map_gt)

        # calculate uncertainty loss
        uncertainty_reg_loss = torch.abs(uncertainty_maps[i][:, 0, :, :] - uncertainty_map_gt[:, 0, :, :]) * valid
        uncertainty_reg_loss = torch.sum(uncertainty_reg_loss, dim=[1, 2])
        uncertainty_reg_loss = uncertainty_reg_loss / (mask_sum + 1e-5)

        # calculate flow prediction loss
        flow_reg_loss = flow_error[:, 0, :, :] * valid * uncertainty_map_gt[:, 0, :, :]
        flow_reg_loss = torch.sum(flow_reg_loss, dim=[1, 2])
        flow_reg_loss = flow_reg_loss / (mask_sum + 1e-5)

        flow_loss += i_weight * (uncertainty_reg_loss.mean() + flow_reg_loss.mean())

        pre_flow_error = flow_error

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics, uncertainty_map_gts



def uncertainty_loss_2(flow_preds, uncertainty_maps, flow_gt, gamma=0.8, MAX_FLOW=400):
    """ 
        Loss function defined from Uncertainty-Depth Joint Optimization Loss Function 
            flow_preds:         LxBx2xHxW
            uncertainty_maps:   LxBx1xHxW
            flow_gt:            Bx2xHxW
    """

    mag = torch.norm(flow_gt, dim=1)  
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask = valid.unsqueeze(1)
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        uncertainty_maps[i] *= 0

        flow_err = torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1, keepdim=True)
        flow_err = torch.exp(-1 * uncertainty_maps[i]) * flow_err
        flow_err += 2 * uncertainty_maps[i]
        flow_err = torch.sum(flow_err * Mask, dim=[1,2,3])
        flow_err = flow_err / (mask_sum + 1e-5)
        flow_loss += i_weight * flow_err.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics    



if __name__ == "__main__":
    flow_preds = torch.tensor([[[[2., 1., 1., 1.],
                                 [4., 1., 1., 1.],
                                 [6., 1., 1., 1.]],

                                [[1., 1., 1., 1.],
                                 [1., 1., 1., 1.],
                                 [1., 1., 1., 1.]]]])
    uncertainty_maps = torch.tensor([[[[1., 1., 1., 1.], 
                                       [1., 1., 1., 1.],
                                       [1., 1., 1., 1.]]]])
    flow_gt = torch.tensor([[[[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]],

                             [[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]]]])

    loss, _ = uncertainty_loss([flow_preds], [uncertainty_maps], flow_gt)