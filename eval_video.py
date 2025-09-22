import os, glob, yaml, cv2, numpy as np, torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from models.matte_unet import MatteUNet
from utils.trimap import weak_trimap_from_rgb
from utils.inference import sliding_window_alpha, temporal_ema

def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('cfg', None)
    in_ch = cfg['model']['in_channels'] if cfg else 4
    model = MatteUNet(in_ch=in_ch).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, cfg

def main(input_dir, ckpt_path, out_dir, ema=0.8, tile=(1024,576), overlap=0.25, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg')))
    assert frames, 'No input frames'
    model, cfg = load_ckpt(ckpt_path, device)
    prev = None
    for fp in tqdm(frames):
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        tri = weak_trimap_from_rgb(bgr)
        img_t = to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        tri_t = torch.from_numpy((tri==255).astype(np.float32) + 0.5*(tri==128).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        alpha = sliding_window_alpha(model, img_t, tri_t, tile=tile, overlap=overlap, device=device)
        alpha_s = alpha if prev is None else temporal_ema(prev, alpha, strength=ema)
        prev = alpha_s
        a8 = (alpha_s.squeeze().clamp(0,1).cpu().numpy()*255.0+0.5).astype(np.uint8)
        base = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(os.path.join(out_dir, f'{base}_alpha.png'), a8)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ema', type=float, default=0.8)
    ap.add_argument('--tile_h', type=int, default=1024)
    ap.add_argument('--tile_w', type=int, default=576)
    ap.add_argument('--overlap', type=float, default=0.25)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()
    main(args.input_dir, args.ckpt, args.out_dir, ema=args.ema, tile=(args.tile_h,args.tile_w), overlap=args.overlap, device=args.device)
