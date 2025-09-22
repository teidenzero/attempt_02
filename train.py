import os, glob, yaml, random, math
import cv2, numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from models.matte_unet import MatteUNet
from losses.matte_losses import matte_loss
from utils.trimap import alpha_to_trimap, weak_trimap_from_rgb

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class FramesAlphaDataset(Dataset):
    def __init__(self, frames_dir, alpha_dir=None, crop=(1024,576), augment=True):
        self.frames = sorted(glob.glob(os.path.join(frames_dir, '*.png')) + glob.glob(os.path.join(frames_dir, '*.jpg')))
        assert self.frames, f'No frames in {frames_dir}'
        self.alpha_dir = alpha_dir
        self.crop = crop
        self.augment = augment

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        imp = self.frames[idx]
        bgr = cv2.imread(imp, cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        alpha = None
        if self.alpha_dir is not None:
            base = os.path.splitext(os.path.basename(imp))[0] + '.png'
            ap = os.path.join(self.alpha_dir, base)
            if os.path.exists(ap):
                a = cv2.imread(ap, cv2.IMREAD_UNCHANGED)
                if a is not None:
                    if a.ndim == 3: a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                    alpha = (a.astype(np.float32) / (65535.0 if a.dtype==np.uint16 else 255.0))
        if alpha is None:
            alpha = np.zeros((h,w), np.float32)

        if (alpha>0).any():
            trimap = alpha_to_trimap((alpha*255).astype(np.uint8), unknown_width=25)
        else:
            trimap = weak_trimap_from_rgb(bgr)

        ch, cw = self.crop
        if h < ch or w < cw:
            bgr = cv2.copyMakeBorder(bgr, 0, max(0, ch-h), 0, max(0, cw-w), cv2.BORDER_REFLECT_101)
            alpha = cv2.copyMakeBorder(alpha, 0, max(0, ch-h), 0, max(0, cw-w), cv2.BORDER_CONSTANT, value=0)
            trimap = cv2.copyMakeBorder(trimap, 0, max(0, ch-h), 0, max(0, cw-w), cv2.BORDER_CONSTANT, value=0)

        y = np.random.randint(0, bgr.shape[0]-ch+1); x = np.random.randint(0, bgr.shape[1]-cw+1)
        bgr = bgr[y:y+ch, x:x+cw]; alpha = alpha[y:y+ch, x:x+cw]; trimap = trimap[y:y+ch, x:x+cw]

        if self.augment and random.random() < 0.5:
            bgr = cv2.flip(bgr, 1); alpha = np.ascontiguousarray(np.flip(alpha, 1)); trimap = cv2.flip(trimap, 1)

        img_t = to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tri_t = torch.from_numpy((trimap==255).astype(np.float32) + 0.5*(trimap==128).astype(np.float32)).unsqueeze(0)
        alp_t = torch.from_numpy(alpha).unsqueeze(0).float()
        return img_t, tri_t, alp_t

def warmup_cosine(step, warmup, total_steps):
    if step < warmup: return step / max(1, warmup)
    p = (step - warmup) / max(1, total_steps - warmup)
    return 0.5*(1+math.cos(math.pi*p))

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get('seed',42))
    device = cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu')

    ds = FramesAlphaDataset(cfg['data']['frames_dir'], cfg['data']['alpha_gt_dir'], crop=tuple(cfg['data']['crop_size']), augment=True)
    dl = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True, drop_last=True)

    model = MatteUNet(in_ch=cfg['model']['in_channels'], widths=tuple(cfg['model']['base_widths']), norm=cfg['model']['norm'], act=cfg['model']['act']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    total_steps = cfg['optim']['epochs'] * max(1, len(dl)); step = 0

    w = (cfg['loss']['w_l1'], cfg['loss']['w_grad'], cfg['loss']['w_lap'], cfg['loss']['w_temp']); lap_levels = cfg['loss']['lap_levels']

    os.makedirs('checkpoints', exist_ok=True); best = 1e9
    for epoch in range(cfg['optim']['epochs']):
        model.train(); from tqdm import tqdm; pbar = tqdm(dl, desc=f'epoch {epoch}')
        for img, tri, alp in pbar:
            img, tri, alp = img.to(device, non_blocking=True), tri.to(device), alp.to(device)
            inp = torch.cat([img, tri], dim=1)

            lr_mult = warmup_cosine(step, cfg['optim']['warmup_steps'], total_steps)
            for pg in opt.param_groups: pg['lr'] = cfg['optim']['lr'] * lr_mult

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                logits = model(inp)
                loss = matte_loss(logits, alp, img, prev=None, weights=w, lap_levels=lap_levels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss.item())); step += 1

        torch.save({'model': model.state_dict(), 'cfg': cfg}, 'checkpoints/last.ckpt')
        if loss.item() < best:
            best = loss.item()
            torch.save({'model': model.state_dict(), 'cfg': cfg}, 'checkpoints/best.ckpt')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
