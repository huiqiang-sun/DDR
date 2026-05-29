import torch
import torch.nn as nn
from torch.nn import functional as F

class ClipIntegral(torch.autograd.Function):
    ''' Clip integral grad
    '''
    AMPLITUDE = 2

    @staticmethod
    def forward(ctx, input, weight):
        assert isinstance(input, torch.Tensor), 'ClipIntegral only takes input as torch.Tensor'
        input_size = input.size()
        ctx.input_size = input_size
        output = input.mul(weight)
        ctx.save_for_backward(input, weight, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # weight: (W)
        # grad_output: (B, K, N, W) or (B, K, W)
        # output: (B, K, N, W) or (B, K, W)
        input, weight, output = ctx.saved_tensors
        output_coord = output.sum(dim=-1, keepdim=True)
        # weight = weight[None, None, :].repeat(
        #     output_coord.shape[0], output_coord.shape[1], 1)
        weight = weight.expand_as(output)

        weight_mask = torch.ones(weight.shape, dtype=grad_output.dtype,
                                 layout=grad_output.layout, device=grad_output.device)
        weight_mask[weight < output_coord] = -1
        weight_mask *= ClipIntegral.AMPLITUDE
        return grad_output.mul(weight_mask), grad_output.mul(input)


def uni2tri(eps):
    # eps U[0, 1]
    # PDF:
    # y = x + 1 (-1 < x < 0)
    # y = -x + 1 (0 < x < 1)
    # CDF:
    # y = x^2 / 2 + x + 1/2 (-1 < x < 0)
    # y = -x^2 / 2 + x + 1/2 (0 < x < 1)
    # invcdf:
    # x = sqrt(2y) - 1, y < 0.5
    # x = 1 - sqrt(2 - 2y), y > 0.5
    tri = torch.where(eps < 0.5, torch.sqrt(2 * eps) - 1, 1 - torch.sqrt(2 - 2 * eps))
    p = torch.where(tri < 0, tri + 1, - tri + 1)
    return tri, p


def retrive_p(hm, x):
    # hm: (B, K, W) or (B, K, S, W)
    # x:  (B, K, W) or (B, K, S, W)
    left_x = x.floor() + 1
    right_x = (x + 1).floor() + 1
    left_hm = F.pad(hm, (1, 1)).gather(-1, left_x.long())
    right_hm = F.pad(hm, (1, 1)).gather(-1, right_x.long())
    new_hm = left_hm + (right_hm - left_hm) * (x + 1 - left_x)

    return new_hm



# weights_map大小为[N_rays, N_samples]
# 输出gumbel_softmax采样结果，大小为[N_rays, num_samples, N_samples]
def norm_weights_map(weights_map, num_samples=1, tau=2):
    shape = weights_map.shape
    if len(shape) != 2:
        print('Error weights shape')
        return
    weights_map = weights_map / (weights_map.sum(dim=1, keepdim=True) + 1e-8)
    weights_map = weights_map.reshape(shape[0], 1, -1) #[N_rays, 1, N_samples]

    eps = torch.rand(weights_map.shape[0], num_samples, weights_map.shape[-1], device=weights_map.device)
    log_eps = torch.log(-torch.log(eps + 1e-8) + 1e-8)
    gumbel_map = (torch.log(weights_map + 1e-8) - log_eps) / tau
    #gumbel_map = (torch.log(weights_map) - log_eps) #/ tau
    gumbel_map = F.softmax(gumbel_map, -1)

    return gumbel_map.reshape(shape[0], num_samples, shape[-1])


# 该函数表示重采样损失，用来约束动态场景的渲染权重的分布与真实深度值类似
# weights表示渲染权重，大小为[N_rays, N_samples]
# target表示真实深度值，大小为[N_rays]
# num_samples表示重采样损失的采样点数量，用于计算损失的期望
# tau表示计算过程中的温度系数，basis_type表示子分部的类型，包括平均分布，三角分布和高斯分布
def sampling_argmax_loss(weights, target, num_samples, tau, basis_type):

    integral = ClipIntegral.apply
    
    norm_weights = norm_weights_map(weights, num_samples, tau) #[N_rays, num_samples, N_samples]
    heat_map = norm_weights / norm_weights.sum(dim=2, keepdim=True)
    N_samples = heat_map.shape[-1]
    
    if basis_type == 'uni':
        w = torch.arange(N_samples, dtype=torch.float32, device=heat_map.device).expand_as(heat_map)
        eps = torch.rand_like(w) - 0.5
        w = w + eps
        location = integral(heat_map, w)
    elif basis_type == 'gaussian':
        w = torch.arange(N_samples, dtype=torch.float32, device=heat_map.device).expand_as(heat_map)
        eps = torch.randn_like(w)

        eps_p = torch.exp(-eps ** 2 * 2)
        heat_map = heat_map * eps_p
        heat_map = heat_map / heat_map.sum(dim=-1, keepdim=True)

        w = w + eps
        location = integral(heat_map, w)
    elif basis_type == 'tri':
        w = torch.arange(N_samples, dtype=torch.float32, device=heat_map.device).expand_as(heat_map)
        eps, _ = uni2tri(torch.rand_like(w))
        w = w + eps

        heat_map = retrive_p(heat_map, w)
        heat_map = heat_map / heat_map.sum(dim=-1, keepdim=True)
        location = integral(heat_map, w)
    else:
        print('Error: Wrong sub-distribution type')
        return
    
    location = location.sum(dim=-1, keepdim=True)
    pred_location = location / float(N_samples) #[N_rays, num_samples, 1]，大小在[0,1]之间

    pred = pred_location.squeeze(-1)
    target = target.expand([num_samples, target.shape[-1]])
    target = target.T

    t_pred = torch.median(pred)
    s_pred = torch.mean(torch.abs(pred - t_pred))

    t_gt = torch.median(target)
    s_gt = torch.mean(torch.abs(target - t_gt))

    pred = (pred - t_pred)/s_pred
    target = (target - t_gt)/s_gt

    loss = torch.abs(pred - target)
    loss = loss.sum() / len(pred)
    loss = loss / num_samples

    return loss