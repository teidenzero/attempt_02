import torch, torch.nn.functional as F

@torch.inference_mode()
def sliding_window_alpha(model, image, trimap, tile=(1024,576), overlap=0.25, device='cuda'):
    model.eval()
    _, _, H, W = image.shape
    th, tw = tile
    sh = max(1, int(th * (1 - overlap)))
    sw = max(1, int(tw * (1 - overlap)))
    out = torch.zeros((1, 1, H, W), device=device)
    norm = torch.zeros((1, 1, H, W), device=device)
    y = 0
    while y < H:
        x = 0
        while x < W:
            patch_img = image[:, :, y:min(y+th, H), x:min(x+tw, W)]
            patch_tri = trimap[:, :, y:min(y+th, H), x:min(x+tw, W)]
            pad_h = th - patch_img.shape[-2]
            pad_w = tw - patch_img.shape[-1]
            if pad_h > 0 or pad_w > 0:
                patch_img = F.pad(patch_img, (0,pad_w,0,pad_h))
                patch_tri = F.pad(patch_tri, (0,pad_w,0,pad_h))
            inp = torch.cat([patch_img, patch_tri], dim=1).to(device)
            logits = model(inp)
            alpha = torch.sigmoid(logits)[..., :patch_img.shape[-2], :patch_img.shape[-1]]
            out[:, :, y:min(y+th,H), x:min(x+tw,W)] += alpha[:, :, :min(th,H-y), :min(tw,W-x)]
            norm[:, :, y:min(y+th,H), x:min(x+tw,W)] += 1.0
            x += sw
        y += sh
    return out / norm.clamp_min(1e-6)

def temporal_ema(prev_alpha, cur_alpha, strength=0.8):
    return strength * prev_alpha + (1.0 - strength) * cur_alpha
