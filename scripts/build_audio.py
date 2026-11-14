import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 44100
# Linear-fit measured exact beat grid of 317RHaFF7Xk.wav (max residual 2.37ms)
BEAT = 0.685650          # 87.508 BPM
DROP_SRC = 11.02288      # Exact downbeat drop in source track
INTRO_BEATS = 6          # 4.1139s intro before drop
T_START = DROP_SRC - INTRO_BEATS * BEAT   # 6.90898s
DROP_VID = INTRO_BEATS * BEAT             # 4.11390s
TOTAL_BEATS = 30
TOTAL_DUR = TOTAL_BEATS * BEAT            # 20.56950s
TOTAL_SAMPS = int(round(TOTAL_DUR * SR))

y_music, sr_m = sf.read('/home/antigravity/work/music/317RHaFF7Xk.wav')
assert sr_m == SR
if y_music.ndim == 1:
    y_music = np.stack([y_music, y_music], axis=1)
y_music = y_music.astype(np.float64)

# Segment 1: 18 beats (Beat 0..18, i.e., T_START -> DROP_SRC + 12*BEAT)
# Segment 2: 12 beats (Beat 18..30, seamlessly continuing from DROP_SRC + 4*BEAT -> DROP_SRC + 16*BEAT)
# Use symmetric overlap around Beat 18 so Beat 18..29 incur ZERO cumulative timing shift!
SPLICE_VID_SAMP = int(round(18 * BEAT * SR))
XF_HALF = int(round(0.015 * SR))  # 30ms equal-power symmetric crossfade
XF_LEN = 2 * XF_HALF

s1_a = int(round(T_START * SR))
s1_len = SPLICE_VID_SAMP + XF_HALF
seg1 = y_music[s1_a : s1_a + s1_len].copy()

s2_center = int(round((DROP_SRC + 4 * BEAT) * SR))
s2_a = s2_center - XF_HALF
s2_len = (TOTAL_SAMPS - SPLICE_VID_SAMP) + XF_HALF
seg2 = y_music[s2_a : s2_a + s2_len].copy()

# Equal-power fade inside [SPLICE_VID_SAMP - XF_HALF : SPLICE_VID_SAMP + XF_HALF]
theta = np.linspace(0, np.pi / 2, XF_LEN)[:, None]
seg1[-XF_LEN:] *= np.cos(theta)
seg2[:XF_LEN] *= np.sin(theta)

music_out = np.zeros((TOTAL_SAMPS, 2), dtype=np.float64)
music_out[:len(seg1)] += seg1
pos2 = SPLICE_VID_SAMP - XF_HALF
avail2 = TOTAL_SAMPS - pos2
music_out[pos2:] += seg2[:avail2]

# Crisp fade-in (0.12s) and smooth exponential outro fade (19.65s -> 20.57s)
fi = int(round(0.12 * SR))
music_out[:fi] *= (np.sin(np.linspace(0, np.pi / 2, fi)) ** 1.2)[:, None]

out_fade_start = int(round(19.65 * SR))
out_len = TOTAL_SAMPS - out_fade_start
music_out[out_fade_start:] *= (np.cos(np.linspace(0, np.pi / 2, out_len)) ** 1.5)[:, None]

# Load external SFX + match crowd roar
y_whoosh, _ = sf.read('/home/antigravity/work/sfx/b9hBHt317mw.wav')
y_boom, _ = sf.read('/home/antigravity/work/sfx/Yzq8c2Xv5IY.wav')
y_crowd, _ = sf.read('/home/antigravity/work/sfx/lille_crowd.wav')


def norm_sfx(arr, a_sec, b_sec, target_peak=0.8):
    s = arr[int(a_sec * SR):int(b_sec * SR)].astype(np.float64)
    if s.ndim == 1:
        s = np.stack([s, s], axis=1)
    e = min(int(0.015 * SR), len(s) // 4)
    s[:e] *= np.linspace(0, 1, e)[:, None]
    s[-e:] *= np.linspace(1, 0, e)[:, None]
    pk = np.max(np.abs(s)) + 1e-9
    return s / pk * target_peak


def place(buf, clip, t_sec, gain=1.0):
    idx = int(round(t_sec * SR))
    if idx < 0:
        clip = clip[-idx:]
        idx = 0
    end = min(len(buf), idx + len(clip))
    if end > idx:
        buf[idx:end] += clip[:end - idx] * gain


whoosh_riser = norm_sfx(y_whoosh, 3.95, 5.15, 0.75)    # riser whoosh
whoosh_short1 = norm_sfx(y_whoosh, 17.45, 18.08, 0.75)  # crisp whip swoosh
whoosh_short2 = norm_sfx(y_whoosh, 18.72, 19.25, 0.75)  # punchy whip swoosh
sub_drop = norm_sfx(y_boom, 0.58, 2.35, 0.85)           # deep cinema boom

# High-pass crowd roar (>300Hz) so stadium energy sits cleanly above 808 bass
sos_hp = butter(4, 300, btype='highpass', fs=SR, output='sos')
crowd_raw = norm_sfx(y_crowd, 8.8, 13.5, 0.7)
crowd_hp = sosfiltfilt(sos_hp, crowd_raw, axis=0)
c_env = np.sin(np.linspace(0, np.pi, len(crowd_hp))) ** 0.75
crowd_hp *= c_env[:, None]

sfx_bus = np.zeros_like(music_out)

# 1. Pre-drop riser building smoothly into Beat 6 (4.1139s)
place(sfx_bus, whoosh_riser, DROP_VID - 0.88, gain=0.40)
# 2. Drop sub-impact hit right at Beat 6 (4.1139s)
place(sfx_bus, sub_drop, DROP_VID - 0.035, gain=0.33)

# 3. Whip transition swooshes aligned cleanly to cut transitions
whip_beats = [2, 8, 11, 14, 16, 18, 20, 23]
for i, b in enumerate(whip_beats):
    t_cut = b * BEAT
    w = whoosh_short1 if i % 2 == 0 else whoosh_short2
    place(sfx_bus, w, t_cut - 0.27, gain=0.23)

# 4. Stadium roar during Bicycle + Lille FK + Celebration climax (Beat 23..29)
place(sfx_bus, crowd_hp, 23 * BEAT, gain=0.28)
# 5. Final cinema slam on Beat 27 (Lille FK hitting net / Roar)
place(sfx_bus, sub_drop, 27 * BEAT - 0.035, gain=0.36)

# Gentle soft-clip limiting & -1.35 dBFS peak headroom for clean AAC conversion
mix = music_out * 0.84 + sfx_bus
pre_pk = np.max(np.abs(mix))
mix = np.tanh(mix * 1.04) / np.tanh(1.04) * 0.855
print(f"Pre-limit peak: {pre_pk:.3f}, Post-limit peak: {np.max(np.abs(mix)):.3f} ({20*np.log10(np.max(np.abs(mix))):.2f} dBFS), Duration: {len(mix)/SR:.5f}s")

sf.write('/home/antigravity/work/master_audio.wav', mix, SR, subtype='PCM_16')
