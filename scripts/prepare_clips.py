import subprocess, os, json, time
import cv2
import numpy as np

OUT_DIR = '/home/antigravity/work/processed_clips'
os.makedirs(OUT_DIR, exist_ok=True)

# (name, src_file, t_start, duration, crop_expr, flow_mult)
SPECS = [
    ("01_walkout",      "rAFP8H1LWYQ.mp4", 7.80,   1.65, "crop=1580:888:120:96",   2),  # 50fps -> 100fps
    ("02_portrait_bg",  "rAFP8H1LWYQ.mp4", 9.90,   2.20, "crop=1500:844:310:15",   1),  # 50fps clean BG
    ("03_curtain",      "F3JeRoKvQbI.mp4", 36.60,  1.40, "crop=1680:945:120:85",   3),  # 25fps -> 75fps
    ("04_back30",       "2pLbqpwpwXc.mp4", 109.70, 1.60, "crop=1680:945:120:160",  3),  # 25fps -> 75fps
    ("05_mancity",      "2pLbqpwpwXc.mp4", 17.45,  1.95, "crop=1600:900:180:110",  4),  # 25fps -> 100fps (slowmo!)
    ("06_roar",         "2pLbqpwpwXc.mp4", 24.95,  1.95, "crop=1600:900:160:110",  4),  # 25fps -> 100fps (slowmo!)
    ("07_ankle_green",  "rAFP8H1LWYQ.mp4", 356.15, 1.40, "crop=1560:878:150:110",  2),  # 50fps -> 100fps
    ("08_ankle_red",    "rAFP8H1LWYQ.mp4", 357.90, 1.85, "crop=1560:878:130:110",  2),  # 50fps -> 100fps
    ("09_stetienne",    "2pLbqpwpwXc.mp4", 87.30,  3.50, "crop=1580:888:170:115",  4),  # 25fps -> 100fps
    ("10_nice_slide",   "lxlXXEy2lAA.mp4", 53.30,  1.30, "crop=1560:878:180:115",  3),  # 29.97fps -> 90fps
    ("11_nantes_cut",   "hfnDGYXFpV4.mp4", 66.35,  1.80, "crop=1560:878:280:95",   4),  # 25fps -> 100fps
    ("12_lens_goal",    "2pLbqpwpwXc.mp4", 43.75,  2.80, "crop=1600:900:170:100",  4),  # 25fps -> 100fps
    ("13_ultras",       "F3JeRoKvQbI.mp4", 199.80, 1.50, "crop=1660:934:140:100",  2),  # 25fps -> 50fps
    ("14_bicycle",      "YXaAOFVFtLM.mp4", 5.55,   1.95, "crop=1520:855:200:112",  4),  # 30fps -> 120fps (hero!)
    ("15_lille_fk",     "lille_fk.mp4",    76.50,  4.00, "crop=1720:968:190:60",   4),  # 25fps -> 100fps
    ("16_sky_outro",    "rAFP8H1LWYQ.mp4", 3.60,   1.35, "crop=1480:832:40:40",    2),  # 50fps -> 100fps
]

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
W, H = 1920, 1080
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

def process_clip(spec):
    t_beg = time.time()
    name, src, t0, dur, crop, mult = spec
    src_p = f'/home/antigravity/work/clips/{src}'
    tmp_p = f'/tmp/raw_{name}.mp4'
    out_p = f'{OUT_DIR}/{name}.mp4'
    pre = max(0.0, t0 - 2.0)
    off = t0 - pre

    subprocess.run([
        'ffmpeg', '-v', 'error', '-y',
        '-ss', f'{pre:.3f}', '-i', src_p,
        '-ss', f'{off:.3f}', '-t', f'{dur:.3f}',
        '-vf', f'{crop},scale=1920:1080:flags=lanczos',
        '-an', '-c:v', 'libx264', '-crf', '9', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
        tmp_p
    ], check=True)

    cap = cv2.VideoCapture(tmp_p)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    os.remove(tmp_p)

    if mult <= 1 or len(frames) < 2:
        target_fps = src_fps
        out_frames = frames
    else:
        target_fps = src_fps * mult
        grays = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (W // 2, H // 2)) for f in frames]
        out_frames = []
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            fw = cv2.resize(dis.calc(grays[i], grays[i + 1], None), (W, H)) * 2.0
            bw = cv2.resize(dis.calc(grays[i + 1], grays[i], None), (W, H)) * 2.0
            out_frames.append(f0)
            for k in range(1, mult):
                a = k / mult
                v0 = cv2.remap(fw, xs - a * 0.5 * fw[:, :, 0], ys - a * 0.5 * fw[:, :, 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                v1 = -cv2.remap(bw, xs + (1.0 - a) * 0.5 * fw[:, :, 0], ys + (1.0 - a) * 0.5 * fw[:, :, 1],
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                v = 0.5 * (v0 + v1)
                w0 = cv2.remap(f0, xs - a * v[:, :, 0], ys - a * v[:, :, 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                w1 = cv2.remap(f1, xs + (1.0 - a) * v[:, :, 0], ys + (1.0 - a) * v[:, :, 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out_frames.append(cv2.addWeighted(w0, 1.0 - a, w1, a, 0.0))
        out_frames.append(frames[-1])

    proc = subprocess.Popen([
        'ffmpeg', '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', f'{target_fps:.6f}',
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '10', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
        out_p
    ], stdin=subprocess.PIPE)
    for fr in out_frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0
    dt = time.time() - t_beg
    print(f"[{name}] {len(frames)}@{src_fps:.1f}fps -> {len(out_frames)}@{target_fps:.1f}fps ({dt:.1f}s)", flush=True)
    return {
        "name": name,
        "frames": len(out_frames),
        "fps": target_fps,
        "duration": len(out_frames) / target_fps,
    }

meta = {}
t_all = time.time()
for spec in SPECS:
    info = process_clip(spec)
    meta[info["name"]] = info

with open(f'{OUT_DIR}/clip_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f"ALL 16 CLIPS PREPARED IN {time.time()-t_all:.1f}s!")
