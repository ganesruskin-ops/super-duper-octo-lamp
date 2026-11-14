import os, sys, json, math, time, subprocess
import numpy as np
import cv2
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.interpolate import PchipInterpolator
import moderngl

W, H = 1920, 1080
FPS = 60
BEAT = 0.685650          # Measured exact BPM = 87.508 BPM
TOTAL_BEATS = 30
TOTAL_DUR = TOTAL_BEATS * BEAT
TOTAL_FRAMES = int(round(TOTAL_DUR * FPS))  # 1234 frames = 20.5667s

CLIP_DIR = '/home/antigravity/work/processed_clips'
ASSET_DIR = '/home/antigravity/work/assets'
AUDIO_PATH = '/home/antigravity/work/master_audio.wav'
OUT_PATH = '/home/antigravity/work/messi_psg_final.mp4'

with open(f'{CLIP_DIR}/clip_meta.json') as f:
    CLIP_META = json.load(f)


class ClipReader:
    def __init__(self, path, preblur=False):
        self.path = path
        self.preblur = preblur
        self.frames = None
        self.fps = 60.0

    def load(self):
        if self.frames is not None:
            return
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        buf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if self.preblur:
                fr = cv2.GaussianBlur(fr, (45, 45), 14.0)
            buf.append(fr)
        cap.release()
        self.frames = buf

    def unload(self):
        self.frames = None

    def get_frame_sec(self, t_sec):
        self.load()
        idx = int(round(t_sec * self.fps))
        idx = max(0, min(len(self.frames) - 1, idx))
        return self.frames[idx]


TIMELINE = [
    ("01_walkout",    0.0,  2.0,  0.05, 1.50, [(0.0,0.0), (0.5,0.42), (1.0,1.0)], (0.0, 0.0),   'zoom', 0, {'zoom0':1.04, 'zoom1':1.18}),
    ("13_ultras",     2.0,  4.0,  0.05, 1.45, [(0.0,0.0), (0.5,0.50), (1.0,1.0)], (1.0, 0.0),   'whip', 1, {'zoom0':1.04, 'zoom1':1.12}),
    ("03_curtain",    4.0,  5.0,  0.05, 1.25, [(0.0,0.0), (0.5,0.45), (1.0,1.0)], (1.0, 0.15),  'whip', 0, {'zoom0':1.02, 'zoom1':1.16}),
    ("04_back30",     5.0,  6.0,  0.05, 1.45, [(0.0,0.0), (0.5,0.55), (1.0,1.0)], (0.0, 0.0),   'zoom', 0, {'zoom0':1.00, 'zoom1':1.28, 'riser':1.0}),
    # === BEAT 6 DROP (4.1139s) ===
    ("05_mancity",    6.0,  8.0,  0.10, 1.80, [(0.0,0.0), (0.36,0.42), (0.64,0.54), (1.0,1.0)], (0.9, -0.3), 'whip', 0, {'zoom0':1.08, 'zoom1':1.03}),
    ("06_roar",       8.0,  9.0,  0.20, 1.05, [(0.0,0.0), (0.5,0.48), (1.0,1.0)], (0.0, 0.9),   'cut',  0, {'zoom0':1.02, 'zoom1':1.14}),
    ("07_ankle_green",9.0, 10.0,  0.15, 1.25, [(0.0,0.0), (0.5,0.42), (1.0,1.0)], (-0.8, 0.2),  'cut',  0, {'zoom0':1.04, 'zoom1':1.12}),
    ("08_ankle_red", 10.0, 11.0,  0.25, 1.65, [(0.0,0.0), (0.45,0.40), (1.0,1.0)], (-1.0, 0.0), 'whip', 0, {'zoom0':1.05, 'zoom1':1.16}),
    ("09_stetienne", 11.0, 13.0,  0.25, 3.25, [(0.0,0.0), (0.35,0.40), (0.65,0.56), (1.0,1.0)], (0.2, 0.95), 'cut', 0, {'zoom0':1.03, 'zoom1':1.12}),
    ("10_nice_slide",13.0, 14.0,  0.10, 1.15, [(0.0,0.0), (0.5,0.46), (1.0,1.0)], (1.0, 0.0),   'whip', 0, {'zoom0':1.04, 'zoom1':1.14}),
    ("11_nantes_cut",14.0, 16.0,  0.55, 1.48, [(0.0,0.0), (0.40,0.42), (0.72,0.68), (1.0,1.0)], (0.0, 0.0),  'zoom', 0, {'zoom0':1.04, 'zoom1':1.16}),
    ("12_lens_goal", 16.0, 18.0,  0.05, 1.68, [(0.0,0.0), (0.50,0.25), (0.72,0.50), (1.0,1.0)], (0.0, -1.0), 'whip', 0, {'zoom0':1.03, 'zoom1':1.15}),
    ("13_ultras",    18.0, 20.0,  0.85, 1.42, [(0.0,0.0), (0.5,0.48), (1.0,1.0)], (-0.9, -0.3), 'whip', 0, {'zoom0':1.14, 'zoom1':1.32}),
    ("14_bicycle",   20.0, 23.0,  0.15, 1.85, [(0.0,0.0), (0.333,0.49), (0.60,0.58), (1.0,1.0)], (0.95, 0.2), 'whip', 0, {'zoom0':1.04, 'zoom1':1.16}),
    ("15_lille_fk",  23.0, 27.0,  0.05, 3.72, [(0.0,0.0), (0.25,0.33), (0.72,0.78), (1.0,1.0)], (0.0, 0.0),  'zoom', 3, {'zoom0':1.02, 'zoom1':1.16}),
    # === BEAT 27 FINAL SUB-DROP SLAM (18.513s) ===
    ("06_roar",      27.0, 29.0,  0.65, 1.72, [(0.0,0.0), (0.30,0.06), (0.62,0.45), (1.0,1.0)], (0.0, -0.5),  'cut',  2, {'zoom0':1.00, 'zoom1':1.07}),
    ("16_sky_outro", 29.0, 30.0,  0.10, 1.20, [(0.0,0.0), (0.5,0.55), (1.0,1.0)], (0.0, 0.0),   'none', 0, {'zoom0':1.06, 'zoom1':1.00}),
]

SEG_DATA = []
for item in TIMELINE:
    name, b0, b1, st0, st1, pts, wdir, wtype, mode, extra = item
    dur = CLIP_META[name]["duration"]
    st0 = min(st0, dur - 0.1)
    st1 = min(st1, dur - 0.02)
    pchip = PchipInterpolator(
        np.array([p[0] for p in pts], dtype=np.float64),
        np.array([p[1] for p in pts], dtype=np.float64),
    )
    SEG_DATA.append({
        "name": name, "b0": b0, "b1": b1, "st0": st0, "st1": st1,
        "pchip": pchip, "wdir": wdir, "wtype": wtype, "mode": mode, "extra": extra
    })


def sample_seg_src_time(seg, t_sec):
    t0, t1 = seg["b0"] * BEAT, seg["b1"] * BEAT
    u = float(np.clip((t_sec - t0) / max(1e-6, t1 - t0), 0.0, 1.0))
    prog = float(seg["pchip"](u))
    return seg["st0"] + prog * (seg["st1"] - seg["st0"]), u


def beat_impulse(t_sec, t_hit, decay=9.0, freq=22.0):
    dt = t_sec - t_hit
    if dt < 0.0 or dt > 0.7:
        return 0.0
    return math.exp(-dt * decay) * math.cos(dt * freq)


VERT_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

PASS1_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_texA;
uniform sampler2D u_texB;
uniform sampler2D u_texMid;
uniform sampler2D u_texFg;

uniform int   u_modeA;
uniform int   u_modeB;
uniform float u_prog;
uniform float u_zoom;
uniform vec2  u_pan;
uniform float u_tilt;

uniform int   u_hasTrans;
uniform float u_transProg;
uniform vec2  u_whipDir;
uniform float u_riserBlur;
uniform float u_shockProg;
uniform float u_chroma;
uniform float u_flash;

vec2 mirrorUV(vec2 uv) {
    vec2 t = abs(mod(uv, 2.0));
    return 1.0 - abs(t - 1.0);
}

vec2 camTransform(vec2 uv, float scale, vec2 offset, float angle) {
    vec2 p = uv - 0.5;
    float c = cos(angle), s = sin(angle);
    p.x *= 1.777778;
    p = mat2(c, -s, s, c) * p;
    p.x /= 1.777778;
    return p / scale + 0.5 + offset;
}

vec3 sampleBGR(sampler2D tex, vec2 uv) {
    return texture(tex, mirrorUV(uv)).bgr;
}

vec4 sampleBGRA(sampler2D tex, vec2 uv) {
    vec4 t = texture(tex, uv);
    return vec4(t.bgr, t.a);
}

// Smooth 11-tap Gaussian-weighted directional/radial motion blur (no stepped ghosts)
vec3 sampleBlurred(sampler2D tex, vec2 uv, vec2 dir, float amt, float radial) {
    if (amt < 0.0008 && radial < 0.0008) {
        return sampleBGR(tex, uv);
    }
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    const int TAPS = 11;
    vec2 radVec = (uv - 0.5) * radial;
    for (int i = 0; i < TAPS; i++) {
        float u = float(i) / float(TAPS - 1);
        float f = u - 0.5;
        float w = exp(-f * f * 5.0);
        vec2 o = dir * amt * f + radVec * u;
        acc += sampleBGR(tex, uv + o) * w;
        wsum += w;
    }
    return acc / wsum;
}

// 3D Spatial AI Cutout Depth Reveal (uses pre-blurred background texture!)
vec3 renderParallax(sampler2D bgTex, vec2 baseUV, float p) {
    vec2 bgUV = camTransform(baseUV, u_zoom * 1.05, u_pan * 0.4 + vec2((p - 0.5) * 0.015, 0.0), u_tilt * 0.5);
    vec3 bg = mix(sampleBGR(bgTex, bgUV) * 0.46, vec3(0.03, 0.07, 0.16), 0.36);

    float midScale = mix(1.16, 0.98, p);
    vec2 midOff = vec2(-0.020 + (p - 0.5) * -0.024, 0.0) + u_pan * 0.65;
    vec2 midUV = camTransform(baseUV, midScale, midOff, 0.0);
    if (midUV.x >= 0.0 && midUV.x <= 1.0 && midUV.y >= 0.0 && midUV.y <= 1.0) {
        vec4 mid = sampleBGRA(u_texMid, midUV);
        float midFade = smoothstep(0.0, 0.22, p);
        bg = mix(bg, mid.rgb, mid.a * midFade * 0.92);
    }

    float fgScale = mix(0.96, 1.08, p);
    vec2 fgOff = vec2((p - 0.5) * 0.018, 0.0) + u_pan;
    vec2 fgUV = camTransform(baseUV, fgScale, fgOff, u_tilt * 0.8);
    if (fgUV.x >= 0.0 && fgUV.x <= 1.0 && fgUV.y >= 0.0 && fgUV.y <= 1.0) {
        vec4 fg = sampleBGRA(u_texFg, fgUV);
        if (fg.a > 0.005) {
            float aL = texture(u_texFg, clamp(fgUV + vec2(0.0022, 0.0), 0.0, 1.0)).a;
            float aR = texture(u_texFg, clamp(fgUV - vec2(0.0022, 0.0), 0.0, 1.0)).a;
            float rim = clamp(fg.a - min(aL, aR), 0.0, 1.0);
            vec3 fgLit = fg.rgb + vec3(0.35, 0.65, 1.0) * rim * 0.34;
            bg = mix(bg, fgLit, fg.a);
        }
    }
    return bg;
}

vec3 sampleScene(sampler2D tex, int mode, vec2 rawUV, vec2 cUV, vec2 blurDir, float blurAmt, float radBlur, float p) {
    if (mode == 1) {
        vec2 shiftedUV = mirrorUV(rawUV + (cUV - camTransform(rawUV, u_zoom, u_pan, u_tilt)));
        return renderParallax(tex, shiftedUV, p);
    }
    vec3 base = sampleBlurred(tex, cUV, blurDir, blurAmt, radBlur);
    // Single-offset lightweight radial chromatic aberration (only 2 extra lookups)
    if (u_chroma > 0.0008 && blurAmt < 0.002) {
        vec2 dirR = (rawUV - 0.5) * u_chroma;
        base.r = sampleBGR(tex, cUV + dirR).r;
        base.b = sampleBGR(tex, cUV - dirR).b;
    }
    if (mode == 3) {
        vec3 lifted = clamp((base - 0.09) / 0.86, 0.0, 1.0);
        base = mix(base, smoothstep(0.0, 1.0, lifted), 0.72);
    }
    return base;
}

void main() {
    vec2 uv = v_uv;

    // Clean outward shockwave ring that NEVER pinches center subject at (0.5, 0.5)
    if (u_shockProg > 0.001 && u_shockProg < 1.0) {
        vec2 aspectDelta = (uv - 0.5) * vec2(1.777778, 1.0);
        float dist = length(aspectDelta);
        float radius = 0.14 + u_shockProg * 1.15;
        float band = 0.055;
        float env = smoothstep(0.0, 0.12, u_shockProg) * (1.0 - u_shockProg);
        float centerSafe = smoothstep(0.06, 0.22, dist);
        float mask = exp(-pow((dist - radius) / band, 2.0)) * centerSafe;
        vec2 pushDir = normalize(aspectDelta + vec2(1e-5)) * vec2(0.5625, 1.0);
        uv -= pushDir * mask * 0.020 * env;
    }

    vec2 camUV = camTransform(uv, u_zoom, u_pan, u_tilt);
    float radBlur = u_riserBlur * 0.10;
    vec3 col = vec3(0.0);

    if (u_hasTrans == 0) {
        col = sampleScene(u_texA, u_modeA, uv, camUV, vec2(0.0), 0.0, radBlur, u_prog);
    } else {
        float tp = clamp(u_transProg, 0.0, 1.0);
        if (u_hasTrans == 1 || u_hasTrans == 3) {
            float slideScale = (u_hasTrans == 1) ? 0.48 : 0.22;
            float blurScale  = (u_hasTrans == 1) ? 0.062 : 0.032;
            float pIn  = pow(tp, 3.0);
            float pOut = pow(1.0 - tp, 3.0);
            float vel  = sin(tp * 3.14159265);
            vec2 uvA = camUV + u_whipDir * pIn  * slideScale;
            vec2 uvB = camUV - u_whipDir * pOut * slideScale;
            vec3 cA = sampleScene(u_texA, u_modeA, uv, uvA, u_whipDir, blurScale * vel, radBlur, (u_modeA == 1) ? 0.96 : u_prog);
            vec3 cB = sampleScene(u_texB, u_modeB, uv, uvB, u_whipDir, blurScale * vel, 0.0,     (u_modeB == 1) ? 0.02 : u_prog);
            float blend = smoothstep(0.38, 0.62, tp);
            col = mix(cA, cB, blend);
        } else {
            float vel = sin(tp * 3.14159265);
            float zA  = mix(1.0, 1.48, pow(tp, 2.5));
            float zB  = mix(0.76, 1.0, 1.0 - pow(1.0 - tp, 2.5));
            vec2 uvA  = (camUV - 0.5) / zA + 0.5;
            vec2 uvB  = (camUV - 0.5) / zB + 0.5;
            vec3 cA = sampleScene(u_texA, u_modeA, uv, uvA, vec2(0.0), 0.0, radBlur + vel * 0.085, (u_modeA == 1) ? 0.96 : u_prog);
            vec3 cB = sampleScene(u_texB, u_modeB, uv, uvB, vec2(0.0), 0.0, vel * 0.070,             (u_modeB == 1) ? 0.02 : u_prog);
            float blend = smoothstep(0.40, 0.60, tp);
            col = mix(cA, cB, blend);
        }
    }

    int activeMode = (u_hasTrans != 0 && u_transProg >= 0.5) ? u_modeB : u_modeA;
    if (activeMode == 2 && u_prog < 0.65) {
        vec2 liveUV = camTransform(uv, u_zoom, u_pan, u_tilt);
        float aLive = sampleBGRA(u_texFg, liveUV).a;
        float outsideSubject = 1.0 - smoothstep(0.25, 0.78, aLive);

        float e1 = clamp(u_prog / 0.36, 0.0, 1.0);
        float e2 = clamp((u_prog - 0.07) / 0.42, 0.0, 1.0);
        if (e1 > 0.0 && e1 < 1.0) {
            float s1 = 1.0 + e1 * 0.24;
            float a1o = sampleBGRA(u_texFg, camTransform(uv, u_zoom * s1, u_pan, u_tilt)).a;
            float a1i = sampleBGRA(u_texFg, camTransform(uv, u_zoom * (s1 * 0.955), u_pan, u_tilt)).a;
            float ring1 = clamp(a1o - a1i * 0.85, 0.0, 1.0) * outsideSubject;
            col += vec3(0.95, 0.16, 0.24) * ring1 * (1.0 - e1) * 1.15;
        }
        if (e2 > 0.0 && e2 < 1.0) {
            float s2 = 1.0 + e2 * 0.30;
            float a2o = sampleBGRA(u_texFg, camTransform(uv, u_zoom * s2, u_pan, u_tilt)).a;
            float a2i = sampleBGRA(u_texFg, camTransform(uv, u_zoom * (s2 * 0.955), u_pan, u_tilt)).a;
            float ring2 = clamp(a2o - a2i * 0.85, 0.0, 1.0) * outsideSubject;
            col += vec3(0.18, 0.62, 1.00) * ring2 * (1.0 - e2) * 0.95;
        }
    }

    col += col * u_flash * 0.40 + vec3(u_flash * 0.08);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

BLOOM_PRE_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;

void main() {
    vec3 c = texture(u_tex, v_uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 streak = c * smoothstep(0.66, 0.96, lum) * 0.22;
    vec2 px = vec2(1.0 / 480.0, 0.0);
    float w[4] = float[](0.18, 0.12, 0.06, 0.03);
    for (int i = 1; i <= 4; i++) {
        vec2 o = px * float(i) * 3.2;
        vec3 sL = texture(u_tex, clamp(v_uv - o, 0.0, 1.0)).rgb;
        vec3 sR = texture(u_tex, clamp(v_uv + o, 0.0, 1.0)).rgb;
        float kL = smoothstep(0.68, 0.96, dot(sL, vec3(0.2126, 0.7152, 0.0722)));
        float kR = smoothstep(0.68, 0.96, dot(sR, vec3(0.2126, 0.7152, 0.0722)));
        streak += (sL * kL + sR * kR) * w[i - 1];
    }
    streak *= vec3(0.88, 0.96, 1.15);
    fragColor = vec4(streak, 1.0);
}
"""

BLOOM_V_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;

void main() {
    vec2 py = vec2(0.0, 1.0 / 270.0);
    float w[4] = float[](0.28, 0.20, 0.11, 0.05);
    vec3 res = texture(u_tex, v_uv).rgb * w[0];
    for (int i = 1; i < 4; i++) {
        res += texture(u_tex, clamp(v_uv + py * float(i) * 2.0, 0.0, 1.0)).rgb * w[i];
        res += texture(u_tex, clamp(v_uv - py * float(i) * 2.0, 0.0, 1.0)).rgb * w[i];
    }
    fragColor = vec4(res, 1.0);
}
"""

FINAL_GRADE_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_time;
uniform float u_bloomAmt;
uniform float u_barFrac;
uniform float u_fade;
uniform float u_bass;

vec3 acesFilmic(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float distEdge = min(v_uv.y, 1.0 - v_uv.y);
    float barMask = smoothstep(u_barFrac - 0.0014, u_barFrac + 0.0014, distEdge);
    if (barMask < 0.001) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = texture(u_scene, v_uv).rgb;
    vec3 blm = texture(u_bloom, v_uv).rgb;

    col = col + blm * u_bloomAmt * (1.0 - col * 0.35);

    float greenExcess = clamp(col.g - max(col.r, col.b) * 0.92, 0.0, 1.0);
    col = mix(col, vec3(col.g * 0.58, col.g * 0.80, col.g * 0.68), greenExcess * 0.55);

    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    vec3 shadowTint = vec3(0.025, 0.055, 0.145);
    vec3 highTint   = vec3(0.98, 1.01, 1.06);
    col = mix(col + shadowTint * 0.35, col * highTint, smoothstep(0.08, 0.78, lum));

    col = pow(max(col, 0.0), vec3(1.06));
    col *= (1.14 + u_bass * 0.06);
    col = acesFilmic(col);

    vec2 vc = (v_uv - 0.5) * vec2(1.18, 1.0);
    float vig = 1.0 - dot(vc, vc) * 0.42;
    col *= clamp(vig, 0.35, 1.0);

    float g = hash12(v_uv * vec2(1920.0, 1080.0) + fract(u_time * 60.0) * 137.0) - 0.5;
    col += g * 0.020;

    col *= barMask * u_fade;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def set_u(p, k, v):
    if k in p:
        p[k] = v


def run_render(start_frame=0, end_frame=TOTAL_FRAMES, out_path=OUT_PATH, lossless_video=False):
    t_start_render = time.time()

    y_aud, sr_a = sf.read(AUDIO_PATH)
    ym = np.mean(y_aud, axis=1)
    sos_sub = butter(4, 125.0, btype='lowpass', fs=sr_a, output='sos')
    sub_sig = sosfilt(sos_sub, ym)

    frame_times = np.arange(TOTAL_FRAMES) / FPS
    bass_env = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    for fi, ft in enumerate(frame_times):
        c = int(ft * sr_a)
        a = max(0, c - 735)
        b = min(len(sub_sig), c + 735)
        bass_env[fi] = np.sqrt(np.mean(sub_sig[a:b] ** 2))
    bass_env = np.clip(bass_env / (np.percentile(bass_env, 98) + 1e-6), 0.0, 1.3)
    for fi in range(1, TOTAL_FRAMES):
        if bass_env[fi] < bass_env[fi - 1]:
            bass_env[fi] = bass_env[fi - 1] * 0.84 + bass_env[fi] * 0.16

    ALL_BEAT_TIMES = [b * BEAT for b in range(6, 30)]
    MAJOR_IMPACTS = [6 * BEAT, 21 * BEAT, 24 * BEAT, 27 * BEAT]

    ctx = moderngl.create_context(standalone=True, backend='egl')
    quad_buf = ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4').tobytes())

    prog_p1 = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=PASS1_FRAG)
    prog_bp = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=BLOOM_PRE_FRAG)
    prog_bv = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=BLOOM_V_FRAG)
    prog_fg = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FINAL_GRADE_FRAG)

    vao_p1 = ctx.simple_vertex_array(prog_p1, quad_buf, 'in_pos')
    vao_bp = ctx.simple_vertex_array(prog_bp, quad_buf, 'in_pos')
    vao_bv = ctx.simple_vertex_array(prog_bv, quad_buf, 'in_pos')
    vao_fg = ctx.simple_vertex_array(prog_fg, quad_buf, 'in_pos')

    tex_fbo1 = ctx.texture((W, H), 4, dtype='f1')
    tex_fbo1.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo1 = ctx.framebuffer(color_attachments=[tex_fbo1])

    tex_bp = ctx.texture((W // 4, H // 4), 4, dtype='f1')
    tex_bp.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_bp = ctx.framebuffer(color_attachments=[tex_bp])

    tex_bv = ctx.texture((W // 4, H // 4), 4, dtype='f1')
    tex_bv.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_bv = ctx.framebuffer(color_attachments=[tex_bv])

    tex_final = ctx.texture((W, H), 4, dtype='f1')
    fbo_final = ctx.framebuffer(color_attachments=[tex_final])

    def make_tex(w, h, comp):
        t = ctx.texture((w, h), comp)
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = False
        t.repeat_y = False
        return t

    texA = make_tex(W, H, 3)
    texB = make_tex(W, H, 3)

    img_mid = cv2.imread(f'{ASSET_DIR}/bg_30_layer.png', cv2.IMREAD_UNCHANGED)
    img_fg_portrait = cv2.imread(f'{ASSET_DIR}/cutout_portrait_centered.png', cv2.IMREAD_UNCHANGED)
    img_fg_roar = cv2.imread(f'{ASSET_DIR}/cutout_roar_aligned.png', cv2.IMREAD_UNCHANGED)

    texMid = make_tex(W, H, 4)
    texMid.write(img_mid.tobytes())
    texFgPortrait = make_tex(W, H, 4)
    texFgPortrait.write(img_fg_portrait.tobytes())
    texFgRoar = make_tex(W, H, 4)
    texFgRoar.write(img_fg_roar.tobytes())

    readers = {}
    def get_reader(name, preblur=False):
        key = (name, preblur)
        if key not in readers:
            loaded = [k for k, v in readers.items() if v.frames is not None]
            if len(loaded) >= 3:
                readers[loaded[0]].unload()
            readers[key] = ClipReader(f'{CLIP_DIR}/{name}.mp4', preblur=preblur)
        return readers[key]

    if lossless_video:
        ff_cmd = [
            'ffmpeg', '-v', 'warning', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', f'{FPS}', '-i', '-',
            '-an', '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'ultrafast',
            out_path
        ]
    else:
        ff_cmd = [
            'ffmpeg', '-v', 'warning', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', f'{FPS}', '-i', '-',
            '-ss', f'{start_frame/FPS:.5f}', '-i', AUDIO_PATH,
            '-c:v', 'libx264', '-crf', '15', '-preset', 'fast', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '256k',
            '-shortest', '-movflags', '+faststart',
            out_path
        ]
    ffmpeg_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    prog_p1['u_texA'] = 0
    prog_p1['u_texB'] = 1
    prog_p1['u_texMid'] = 2
    prog_p1['u_texFg'] = 3
    prog_bp['u_tex'] = 4
    prog_bv['u_tex'] = 5
    prog_fg['u_scene'] = 4
    prog_fg['u_bloom'] = 5
    texMid.use(2)

    WHIP_HALF_SEC = 0.068
    CUT_HALF_SEC  = 0.025

    for fi in range(start_frame, end_frame):
        t_sec = fi / FPS

        seg_idx = len(SEG_DATA) - 1
        for i, s in enumerate(SEG_DATA):
            if s["b0"] * BEAT <= t_sec < s["b1"] * BEAT:
                seg_idx = i
                break
        seg = SEG_DATA[seg_idx]
        src_t_a, seg_prog = sample_seg_src_time(seg, t_sec)
        frA = get_reader(seg["name"], preblur=(seg["mode"] == 1)).get_frame_sec(src_t_a)

        modeA = seg["mode"]
        modeB = seg["mode"]
        has_trans = 0
        trans_prog = 0.0
        whip_dir = (0.0, 0.0)
        t_end = seg["b1"] * BEAT
        t_start = seg["b0"] * BEAT

        texA_frame = frA
        texB_frame = None

        if seg_idx + 1 < len(SEG_DATA) and seg["wtype"] != 'none':
            half_w = CUT_HALF_SEC if seg["wtype"] == 'cut' else WHIP_HALF_SEC
            if t_sec >= t_end - half_w:
                nxt = SEG_DATA[seg_idx + 1]
                src_t_b, _ = sample_seg_src_time(nxt, t_sec)
                texB_frame = get_reader(nxt["name"], preblur=(nxt["mode"] == 1)).get_frame_sec(src_t_b)
                modeB = nxt["mode"]
                has_trans = {'whip': 1, 'zoom': 2, 'cut': 3}[seg["wtype"]]
                trans_prog = 0.5 * (t_sec - (t_end - half_w)) / half_w
                whip_dir = seg["wdir"]
        if has_trans == 0 and seg_idx > 0 and SEG_DATA[seg_idx - 1]["wtype"] != 'none':
            prv = SEG_DATA[seg_idx - 1]
            half_w = CUT_HALF_SEC if prv["wtype"] == 'cut' else WHIP_HALF_SEC
            if t_sec < t_start + half_w:
                src_t_prev, _ = sample_seg_src_time(prv, t_sec)
                texA_frame = get_reader(prv["name"], preblur=(prv["mode"] == 1)).get_frame_sec(src_t_prev)
                texB_frame = frA
                modeA = prv["mode"]
                modeB = seg["mode"]
                has_trans = {'whip': 1, 'zoom': 2, 'cut': 3}[prv["wtype"]]
                trans_prog = 0.5 + 0.5 * (t_sec - t_start) / half_w
                whip_dir = prv["wdir"]

        texA.use(0)
        texA.write(texA_frame.tobytes())
        if texB_frame is not None:
            texB.use(1)
            texB.write(texB_frame.tobytes())

        if modeA == 2 or modeB == 2:
            texFgRoar.use(3)
        else:
            texFgPortrait.use(3)

        ex = seg["extra"]
        z0, z1 = ex.get("zoom0", 1.02), ex.get("zoom1", 1.12)
        base_zoom = z0 + (z1 - z0) * seg_prog

        punch_zoom = 0.0
        shake_x = 0.0025 * math.sin(t_sec * 3.1)
        shake_y = 0.0020 * math.cos(t_sec * 2.3)
        tilt = 0.006 * math.sin(t_sec * 1.9)
        chroma = 0.0018 + 0.0022 * float(bass_env[fi])
        flash = 0.0
        shock_prog = 0.0

        for bt in ALL_BEAT_TIMES:
            imp = beat_impulse(t_sec, bt, decay=11.0, freq=26.0)
            if abs(imp) > 0.001:
                punch_zoom += 0.022 * max(0.0, imp)
                shake_y += 0.004 * imp
                chroma += 0.003 * max(0.0, imp)

        for mt in MAJOR_IMPACTS:
            dt = t_sec - mt
            if 0.0 <= dt <= 0.42:
                shock_prog = max(shock_prog, dt / 0.42)
                imp = beat_impulse(t_sec, mt, decay=9.5, freq=30.0)
                punch_zoom += 0.045 * max(0.0, imp)
                shake_x += 0.008 * math.sin(dt * 55.0) * math.exp(-dt * 12.0)
                shake_y += 0.008 * math.cos(dt * 48.0) * math.exp(-dt * 12.0)
                tilt += 0.015 * imp
                chroma += 0.0065 * math.exp(-dt * 12.0)
                flash = max(flash, 0.34 * math.exp(-dt * 24.0))

        riser_blur = (seg_prog ** 2) if "riser" in ex else 0.0

        DROP_T = 6 * BEAT
        if t_sec < DROP_T - 0.03:
            bar_frac = 0.128
        elif t_sec < 29 * BEAT:
            dt_d = max(0.0, t_sec - (DROP_T - 0.03))
            snap = math.exp(-dt_d * 10.0) * math.cos(dt_d * 18.0) if dt_d < 0.5 else 0.0
            bar_frac = 0.044 + 0.084 * max(0.0, snap) - 0.005 * float(bass_env[fi])
        else:
            u_end = float(np.clip((t_sec - 29 * BEAT) / BEAT, 0.0, 1.0))
            bar_frac = 0.044 + (0.50 - 0.044) * (u_end ** 1.8)

        fade = 1.0
        if t_sec < 0.32:
            fade = t_sec / 0.32
        elif t_sec > TOTAL_DUR - 0.55:
            fade = max(0.0, (TOTAL_DUR - t_sec) / 0.55)

        bloom_amt = 0.34 + 0.14 * float(bass_env[fi]) + (0.20 if shock_prog > 0 else 0.0)

        fbo1.use()
        ctx.viewport = (0, 0, W, H)
        set_u(prog_p1, 'u_modeA', modeA)
        set_u(prog_p1, 'u_modeB', modeB)
        set_u(prog_p1, 'u_prog', seg_prog)
        set_u(prog_p1, 'u_zoom', base_zoom + punch_zoom)
        set_u(prog_p1, 'u_pan', (shake_x, shake_y))
        set_u(prog_p1, 'u_tilt', tilt)
        set_u(prog_p1, 'u_hasTrans', has_trans)
        set_u(prog_p1, 'u_transProg', trans_prog)
        set_u(prog_p1, 'u_whipDir', whip_dir)
        set_u(prog_p1, 'u_riserBlur', riser_blur)
        set_u(prog_p1, 'u_shockProg', shock_prog)
        set_u(prog_p1, 'u_chroma', chroma)
        set_u(prog_p1, 'u_flash', flash)
        vao_p1.render(moderngl.TRIANGLE_STRIP)

        fbo_bp.use()
        ctx.viewport = (0, 0, W // 4, H // 4)
        tex_fbo1.use(4)
        vao_bp.render(moderngl.TRIANGLE_STRIP)

        fbo_bv.use()
        tex_bp.use(5)
        vao_bv.render(moderngl.TRIANGLE_STRIP)

        fbo_final.use()
        ctx.viewport = (0, 0, W, H)
        tex_fbo1.use(4)
        tex_bv.use(5)
        set_u(prog_fg, 'u_time', t_sec)
        set_u(prog_fg, 'u_bloomAmt', bloom_amt)
        set_u(prog_fg, 'u_barFrac', bar_frac)
        set_u(prog_fg, 'u_fade', fade)
        set_u(prog_fg, 'u_bass', float(bass_env[fi]))
        vao_fg.render(moderngl.TRIANGLE_STRIP)

        ffmpeg_proc.stdin.write(fbo_final.read(components=4))

        total_to_render = end_frame - start_frame
        done = fi - start_frame + 1
        if done % 60 == 1 or fi == end_frame - 1:
            el = time.time() - t_start_render
            print(f"Frame {fi:4d}/{end_frame} ({t_sec:5.2f}s) | {el:5.1f}s elapsed ({done/max(0.1,el):5.2f} fps)", flush=True)

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    assert ffmpeg_proc.returncode == 0
    print(f"RENDER COMPLETE in {time.time()-t_start_render:.1f}s -> {out_path}")


if __name__ == '__main__':
    ll = ('--lossless' in sys.argv)
    if '--bench' in sys.argv:
        run_render(start_frame=0, end_frame=120, out_path='/tmp/bench_montage.mp4')
    elif '--range' in sys.argv:
        idx = sys.argv.index('--range')
        s_f = int(sys.argv[idx + 1])
        e_f = int(sys.argv[idx + 2])
        o_p = sys.argv[idx + 3]
        run_render(start_frame=s_f, end_frame=e_f, out_path=o_p, lossless_video=ll)
    else:
        run_render(lossless_video=ll)
