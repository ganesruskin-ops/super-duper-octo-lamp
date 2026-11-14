import subprocess, json
import cv2
import numpy as np
import soundfile as sf
import librosa

VIDEO = '/home/antigravity/work/messi_psg_final.mp4'
BEAT = 0.685650

# 1. FFprobe verification
probe = subprocess.check_output([
    'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', VIDEO
], text=True)
info = json.loads(probe)
for s in info['streams']:
    print(f"Stream {s['index']} ({s['codec_type']}): codec={s.get('codec_name')}, "
          f"res={s.get('width')}x{s.get('height')}, r_fps={s.get('r_frame_rate')}, "
          f"sr={s.get('sample_rate')}, dur={s.get('duration')}")
print(f"Format duration={info['format']['duration']}, size={int(info['format']['size'])/1e6:.2f} MB")

# 2. Rendered AAC Audio EBU R128 + Beat Onset Verification
tmp_wav = '/tmp/rendered_audio_check.wav'
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', VIDEO, '-vn', '-ar', '44100', tmp_wav], check=True)
y, sr = sf.read(tmp_wav)
ym = np.mean(y, axis=1)
pk = np.max(np.abs(y))
print(f"Rendered Audio Peak={pk:.4f} ({20*np.log10(pk):.2f} dBFS), Clipping samples={np.sum(np.abs(y)>=0.999)}")
ebur = subprocess.run([
    'ffmpeg', '-hide_banner', '-i', VIDEO, '-af', 'ebur128=peak=true', '-f', 'null', '-'
], capture_output=True, text=True).stderr
for line in ebur.splitlines()[-14:]:
    if line.strip():
        print(line)

# 3. Video Frame Flux vs Audio Beat Onset Alignment (streaming to keep RAM tiny!)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
prev_g = None
flux = []
mean_luma = []

target_sb_frames = {min(1233, int(round(k * 0.50 * fps))): k for k in range(42)}
checks = [
    ("parallax_2.0s", 2.05),
    ("riser_3.9s", 3.92),
    ("drop_4.12s", 4.13),
    ("mancity_4.8s", 4.82),
    ("ankle_7.1s", 7.15),
    ("lens_11.5s", 11.50),
    ("bicy_hit_14.4s", 14.42),
    ("lille_hit_16.5s", 16.50),
    ("lille_post_18.4s", 18.42),
    ("roar_echo_18.6s", 18.62),
    ("roar_echo_18.8s", 18.82),
    ("outro_20.0s", 20.05),
]
target_hero_frames = {}
for lbl, ts in checks:
    fi = min(1233, int(round(ts * fps)))
    target_hero_frames.setdefault(fi, []).append(lbl)

sb_thumbs = {}
hero_thumbs = {}
idx_f = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
    mean_luma.append(float(np.mean(g)))
    if prev_g is None:
        flux.append(0.0)
    else:
        flux.append(float(np.mean(np.abs(g - prev_g))))
    prev_g = g

    if idx_f in target_sb_frames:
        sb_thumbs[idx_f] = cv2.resize(f, (480, 270))
    if idx_f in target_hero_frames:
        hero_thumbs[idx_f] = cv2.resize(f, (640, 360))
    idx_f += 1
cap.release()
flux = np.array(flux)
mean_luma = np.array(mean_luma)
print(f"Total video frames read: {idx_f} (@ {fps:.2f}fps)")

# Check zero mid-video black frames (excluding intro fade <0.35s and outro >19.95s)
mid_min_luma = float(np.min(mean_luma[int(0.4*fps):int(19.9*fps)]))
print(f"Mid-video min mean brightness: {mid_min_luma:.2f} (healthy if > 10)")

print("\nVisual Cut/Flux Spikes around Key Beats:")
for b in [2, 4, 6, 8, 11, 14, 16, 18, 20, 21, 23, 24, 27, 29]:
    tb = b * BEAT
    fc = int(round(tb * fps))
    win = flux[max(0, fc - 6):min(len(flux), fc + 7)]
    pk_off = int(np.argmax(win)) - min(6, fc)
    print(f"  Beat {b:2d} ({tb:5.2f}s / f={fc:4d}): peak visual flux {np.max(win):5.2f} at offset {pk_off:+d} frames ({pk_off/fps*1000:+.1f}ms)")

# 4. Generate 42-frame Storyboard (every 0.50s)
thumbs = []
for k in range(42):
    fi = min(idx_f - 1, int(round(k * 0.50 * fps)))
    th = sb_thumbs[fi].copy()
    cv2.putText(th, f'{k*0.50:4.2f}s (f{fi})', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    thumbs.append(th)
grid = np.vstack([np.hstack(thumbs[r*6:(r+1)*6]) for r in range(7)])
cv2.imwrite('/home/antigravity/work/inspect/audit_storyboard.jpg', grid)

# 5. Generate Critical Seam & Effect Strip (Drop, 3D Parallax, Ankle, Bicycle, Lille, Roar Echo)
s_imgs = []
for lbl, ts in checks:
    fi = min(idx_f - 1, int(round(ts * fps)))
    im = hero_thumbs[fi].copy()
    cv2.putText(im, lbl, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    s_imgs.append(im)
s_grid = np.vstack([np.hstack(s_imgs[r*4:(r+1)*4]) for r in range(3)])
cv2.imwrite('/home/antigravity/work/inspect/audit_hero_frames.jpg', s_grid)
print("Saved audit_storyboard.jpg and audit_hero_frames.jpg")
