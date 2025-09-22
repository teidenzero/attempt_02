import cv2, numpy as np

def alpha_to_trimap(alpha_u8, fg_thr=240, bg_thr=15, unknown_width=25):
    fg = (alpha_u8 >= fg_thr).astype(np.uint8)
    bg = (alpha_u8 <= bg_thr).astype(np.uint8)
    unk = 1 - np.clip(fg + bg, 0, 1)
    if unknown_width > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (unknown_width, unknown_width))
        fg_erode = cv2.erode(fg, kernel)
        bg_erode = cv2.erode(bg, kernel)
        unk = 1 - np.clip(fg_erode + bg_erode, 0, 1)
        fg = (fg_erode * (1 - unk)).astype(np.uint8)
        bg = (bg_erode * (1 - unk)).astype(np.uint8)
    trimap = np.zeros_like(alpha_u8, dtype=np.uint8)
    trimap[bg==1] = 0
    trimap[unk==1] = 128
    trimap[fg==1] = 255
    return trimap

def weak_trimap_from_rgb(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    unk = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=1)
    trimap = np.zeros_like(g, dtype=np.uint8)
    trimap[unk>0] = 128
    return trimap
