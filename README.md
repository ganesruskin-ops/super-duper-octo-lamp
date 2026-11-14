# Messi PSG (#30) — 1080p60 Beat-Synced Montage Pipeline

Procedural GPU shader + optical-flow PSG-era (2021–2023) Lionel Messi highlight montage (`20.57s`, `1920x1080 @ 60fps`, `1234` frames).

> Final rendered `.mp4` binaries are distributed via [GitHub Releases](https://github.com/ganesruskin-ops/super-duper-octo-lamp/releases).

## Pipeline Architecture

1. **Master Audio & Beat Grid ([`scripts/build_audio.py`](scripts/build_audio.py))**
   - Linear-regressed beat grid locked at **`BEAT = 0.685650s`** (`87.508 BPM`) over clean instrumental audio (`INTERWORLD - METAMORPHOSIS` intro + main drop with zero vocals).
   - Zero-shift 30 ms equal-power splice at Beat 18 (`12.3417s`).
   - Layered external risers/whip whooshes synced to whip transitions, cinema sub-drops at Beat 6 (`4.1139s`) & Beat 27 (`18.5126s`), HPF (>300 Hz) match stadium roar across Beats 23–29, and soft-tanh limiter (`-0.9 dBFS` true peak).

2. **Optical-Flow Slow-Motion Prep ([`scripts/prepare_clips.py`](scripts/prepare_clips.py))**
   - Clean crop extraction across 16 PSG match moments (eliminating corner watermarks/channel bugs).
   - Bidirectional midpoint-warped **OpenCV `DISOpticalFlow`** (`cv2.DISOPTICAL_FLOW_PRESET_MEDIUM`) interpolating 25/30/50 fps clips up to **100–120 fps** for ghost-free slow motion on curlers, ankle-breakers, the Clermont bicycle kick, and the Lille 95' free kick.

3. **AI Cutouts & Architectural Depth Layer ([`scripts/prep_assets.py`](scripts/prep_assets.py), [`assets/bg_30_layer.html`](assets/bg_30_layer.html))**
   - AI portrait/roar background segmentation followed by 1 px morphological edge erosion, Gaussian feathering, and color dilation decontamination.
   - Transparent brushed titanium **`30`** numeral + PSG Hechter crimson/navy vertical light columns rendered via headless Chromium into RGBA (`bg_30_layer.png`).

4. **4-Pass ModernGL GLSL Compositor ([`scripts/render_montage.py`](scripts/render_montage.py))**
   - **Pass 1 (Scene + Camera + Transitions)**: Monotonic PCHIP speed-ramp splines locking strikes/kicks directly to beats; 11-tap Gaussian-weighted directional whip pans & zoom rushes with mirror-UV wrapping; center-safe radial impact shockwave rings; 3-layer 3D Z-parallax reveal (`S02`); Beat 27 expanding PSG crimson & electric cyan hollow silhouette aura rings (`S16`); de-haze tone curve on foggy Lille stadium footage.
   - **Passes 2–3 (Anamorphic Bloom)**: Quarter-resolution (`480x270`) separable horizontal sapphire streak bloom.
   - **Pass 4 (ACES Filmic Grade + Framing)**: Midnight-navy / emerald-turf split toning, subtle 35 mm film grain, bass-reactive gain, and dynamic `2.39:1 -> 1.88:1` cinema letterbox bars.
   - Supports chunked bit-exact lossless `libx264rgb` (`-crf 0`) range rendering (`--range START END OUT --lossless`) before final H.264 High Profile + 256 kbps AAC muxing.

5. **Quantitative & Visual Audit ([`scripts/audit_output.py`](scripts/audit_output.py))**
   - Verifies stream integrity, post-AAC EBU R128 loudness, zero clipping, mid-video luminance floor, visual frame flux spike offsets per beat, and extracts 42-frame storyboard + hero impact grids.

## Verified Output Metrics (`messi-psg-30-1080p60.mp4`)

| Metric | Measured Value |
| --- | --- |
| Resolution / Frame Rate | `1920x1080` @ `60.00 fps` (`1234` frames) |
| Duration | `20.570 s` (`30` beats @ `87.508 BPM`) |
| Loudness (EBU R128 post-AAC) | `-10.3 LUFS` integrated, `5.8 LU` LRA, `-0.9 dBFS` true peak (`0` clipped samples) |
| Beat Onset Accuracy (Beats 6–29) | Mean abs error **`1.73 ms`**, Max abs error **`4.70 ms`** |
| Visual Cut Flux Lock | `0` to `+1` frame (`0.0 ms` to `+16.7 ms`) across all beat transitions |
| Video / Audio Codecs | H.264 High (`crf 15`, `yuv420p`) + AAC-LC `256 kbps` (`44.1 kHz` stereo) |
