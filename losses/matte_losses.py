import torch
import torch.nn.functional as F

def gradient_loss(alpha, alpha_gt):
    dx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    return (dx(alpha)-dx(alpha_gt)).abs().mean() + (dy(alpha)-dy(alpha_gt)).abs().mean()

def laplacian_pyramid_loss(alpha, alpha_gt, levels=3):
    def blur(x):
        return F.avg_pool2d(x, 3, stride=1, padding=1)
    def lap(x):
        return x - blur(x)
    loss = 0.0
    a, g = alpha, alpha_gt
    for _ in range(levels):
        loss = loss + (lap(a) - lap(g)).abs().mean()
        a = F.avg_pool2d(a, 2, ceil_mode=True)
        g = F.avg_pool2d(g, 2, ceil_mode=True)
    return loss

def temporal_consistency_loss(alpha_t, alpha_tm1, img_t, img_tm1):
    eps = 1e-6
    x = F.normalize(img_t, dim=1)
    y = F.normalize(img_tm1, dim=1)
    w = (x*y).sum(1, keepdim=True)
    w = (w + 1.0) / 2.0
    return (w * (alpha_t - alpha_tm1).abs()).mean()

def matte_loss(alpha_logits, alpha_gt, image, prev=None, weights=(0.5,0.25,0.25,0.2), lap_levels=3):
    alpha = torch.sigmoid(alpha_logits)
    w_l1, w_grad, w_lap, w_temp = weights
    l = 0.0
    l = l + w_l1  * (alpha - alpha_gt).abs().mean()
    l = l + w_grad* gradient_loss(alpha, alpha_gt)
    l = l + w_lap * laplacian_pyramid_loss(alpha, alpha_gt, levels=lap_levels)
    if prev is not None and w_temp > 0:
        l = l + w_temp * temporal_consistency_loss(alpha, prev['alpha'], image, prev['image'])
    return l
