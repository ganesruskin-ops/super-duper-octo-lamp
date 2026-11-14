import cv2
import numpy as np
from PIL import Image

def clean_cutout(in_path, out_path):
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 4
    bgr = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    # Erode slightly to kill bright 1-2px background halo, then gentle Gaussian feather
    a_u8 = (alpha * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(a_u8, kernel, iterations=1)
    soft = cv2.GaussianBlur(eroded.astype(np.float32) / 255.0, (3, 3), 0.7)
    # Tighten core alpha so body stays 100% solid
    soft = np.clip((soft - 0.12) / 0.82, 0.0, 1.0)

    # Color decontamination: replace border pixels with nearby interior pixel colors
    solid_mask = (soft > 0.85).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(1 - solid_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # Simple local dilation color propagation for semi-transparent edge pixels
    dil_bgr = bgr.copy()
    for _ in range(3):
        k = np.ones((3, 3), np.uint8)
        m = (soft > 0.85)
        mx = cv2.dilate(dil_bgr, k)
        edge = (soft > 0.01) & (soft <= 0.85)
        dil_bgr[edge] = mx[edge]

    out = np.dstack([np.clip(dil_bgr, 0, 255).astype(np.uint8), (soft * 255).astype(np.uint8)])
    cv2.imwrite(out_path, out)
    print(f"Cleaned cutout saved: {out_path}")

clean_cutout('/home/antigravity/work/assets/cutout_portrait.png', '/home/antigravity/work/assets/cutout_portrait_clean.png')
clean_cutout('/home/antigravity/work/assets/cutout_roar.png', '/home/antigravity/work/assets/cutout_roar_clean.png')
