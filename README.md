# OpenLeap

A fully open-source driver and hand-tracking stack for the original **Leap
Motion Controller** — no `leapd`, no closed SDK, no dead vendor services.

Named in tribute to the [2013 OpenLeap initiative](https://github.com/openleap/OpenLeap),
which set out to build an open driver for this hardware and stopped at raw
image access. This is an independent implementation (no shared code) that
completes the mission: device bring-up, calibrated stereo capture, 21-joint
hand skeletons, and 3D triangulation in millimeters — all open.

**Why:** Ultraleap's `repo.ultraleap.com` is dead, the closed Gemini stack is
unmaintained, and the original LMC is otherwise e-waste. This stack keeps the
hardware alive — and feeds hand-gesture input to the
[niri](https://github.com/niri-wm/niri) Wayland compositor (the project this
grew out of).

## Layout

| path | what |
|---|---|
| `rust-driver/` | Rust USB driver (libusb/`rusb`): device bring-up, stereo IR streaming, exposure/LED control, calibration reader. Also exports the `leap_open_*` C ABI as `libopenleap.a` for the Monado driver |
| `tracking/` | Python hand-tracking pipeline (Mercury ONNX inference, live GTK4 viewer, stereo calibration/triangulation) — now the reference/validation oracle, not the product (see `ARCHITECTURE.md`) |
| `monado-driver/` | design/reference docs for the Monado `leap_open` driver — the engine path Leap → Mercury (the built C driver lives in the monado fork) |
| `reverse-engineering/` | the protocol recon that made the driver possible (usbmon capture analysis, decoded init sequence) — historical record, nothing runs from here |
| `hand-tracking-models/` | Monado's openly-licensed (CC-BY-SA 4.0) Mercury ONNX models — separate git clone, see Setup |
| `run_live.sh` | one-command live viewer (our Python pipeline) |
| `run_monado.sh` | one-command Leap → Mercury (Monado) harness launcher |
| `ARCHITECTURE.md` / `LICENSING.md` | stack decision (why Mercury) and the Ultraleap-EULA posture |

## How the driver works (`rust-driver/`)

The LMC (USB `f182:0003`, firmware 1.7.0) is a UVC-*shaped* device that does
not work with standard UVC drivers — it boots inert and needs a vendor
bring-up. There is **no crypto handshake**: it just needs the right control
writes. Everything below was decoded from a usbmon capture of the closed
`leapd` talking to the device (see `reverse-engineering/FINDINGS.md`).

1. **Claim** — detach `uvcvideo` from interface 0 (VideoControl, owns the
   vendor extension units) and interface 1 (VideoStreaming, owns bulk IN
   endpoint `0x83`); needs root.
2. **Bring-up** — replay 358 UVC control writes (`bringup_seq.txt`) to the
   extension units: sensor config, LED driver init, UVC probe/commit. After
   this the IR LEDs strobe and the sensor streams.
3. **Streaming** — bulk endpoint `0x83` delivers UVC payloads: 16380 bytes =
   12-byte header (`hlen`, `bmHeaderInfo` with FID/EOF/EOH bits) + 16368
   pixel bytes. Frames are delimited by the EOF bit, with FID-toggle recovery
   when an EOF payload is lost. One frame = **307200 bytes**: 640×240 per eye,
   the two eyes interleaved column-by-column into 1280×240, 8-bit gray,
   ~112 fps (≈ USB 2.0 saturation).
4. **Robust capture architecture** — a dedicated drain thread (SCHED_FIFO
   real-time priority) does nothing but service the endpoint into a one-slot
   mailbox; a writer thread ships the latest frame to stdout and drops frames
   if the consumer is slow. The endpoint is never blocked on a pipe — torn
   frames went from >50% to ~1% with this design.
5. **Device controls** (UVC `SET_CUR`, `bmRequestType 0x21, bRequest 0x01`):
   - **exposure**: Camera Terminal (unit 2) selector `0x0b`, 4-byte µs LE.
     Effective range ~100–1000 µs (≥1000 is near-saturation).
   - **IR LEDs**: Processing Unit (unit 5) selector `0x03`, value
     `sub | (on<<6)`, sub 2/3/4 = left/center/right LED.
   - **calibration**: 156 bytes via "property knocking" — write an address to
     PU sharpness (sel `0x08`), read a byte back from saturation (sel `0x07`).
     Contains per-eye focal/distortion + the 40.01 mm stereo baseline.

```
openleap stream [seq]   # bring up + stream LFRM-framed frames to stdout
  stdin commands:  exp <us> | ae <0|1> | leds <0-7 mask>
openleap info | bringup | calib
```

Stream format: `"LFRM"` + u32 LE frame counter + 307200 raw bytes. Stderr
gets a `diag:` line every 128 frames (brightness, exposure, LED mask, and
frame-integrity counters: dark/short/forced/drop/timeouts/errors).

## How the tracking works (`tracking/`)

Per eye, per frame (`skeleton.py`, conventions ported from Monado Mercury's
`hg_model.cpp` — see header comments for the exact mapping):

1. **Detect** — `grayscale_detection_160x160.onnx` on the letterboxed full
   frame; 2 hand slots, thresholds gate noise (HAND_MIN 0.40).
2. **Crop** — square box around each detection (`crop_scale` × detector size),
   resized to 128×128.
3. **Handedness** — the keypoint net knows ONE handedness; the other hand's
   crop must be mirrored. Auto-calibrated per track by trying both and
   keeping the higher-confidence orientation (fixes the thumb landing on the
   pinky side).
4. **Keypoints** — `grayscale_keypoint_jan18.onnx`: 21 heatmaps (22×22),
   argmax + center-of-mass refine → joint pixels. The previous frame's joints
   feed back as `lastKeypoints` (Mercury's temporal prior, large stability
   win).
5. **Coast** — the detector only *acquires* (it dies past ~150 mm: a hand is
   ~12 px in its squashed view). Tracked hands keep going by cropping around
   the previous joints, with the box re-fit to the keypoint spread each
   frame; survives to ~300 mm+. Green box = detector-confirmed, amber =
   coasting.
6. **Smooth** — per-joint One-Euro filter, confidence-weighted
   (still-hand jitter ~1.5 px).
7. **Triangulate** (both-eyes mode) — stereo-rectified joints from both eyes
   → per-joint disparity → depth (`Z = fx·B/d`, B = 40.01 mm from the device
   calibration) → 21 joints in millimeters. Gates: per-eye confidence,
   epipolar row agreement, plausible depth. Validated against a ruler.

`calibration.py` parses the on-device 156-byte calibration (leapuvc layout),
inverts Leap's radial distortion model into OpenCV coefficients via curve
fit, and builds stereo-rectified undistortion maps (`calib.npz`).

`live_viewer.py` is the debug/tuning GUI: live feed + skeleton overlay,
eye/rotation/flip/undistort controls, IR LED + exposure + AE device controls,
temporal/skeleton toggles, frame-integrity stats, 3D pinch/depth readout, and
a **record** mode that saves model inputs + outputs for offline analysis.

## Setup

```bash
# deps: rust, python3, gtk4 + pygobject (system), polkit
git clone https://github.com/julianjc84/OpenLeap.git
cd OpenLeap
git clone https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models.git
python -m venv --system-site-packages .venv
.venv/bin/pip install onnxruntime opencv-python-headless numpy scipy
( cd rust-driver && cargo build )
./run_live.sh        # one polkit prompt: stops any closed leapd, claims the device
```

The closed `ultraleap-hand-tracking-service` must not hold the device;
`run_live.sh` stops it for the session.

### The Mercury path (full 3D skeleton) — pairs with the monado fork

`run_live.sh` runs *our* Python pipeline. For the real engine — Monado's
**Mercury** hand tracker fed by the `leap_open` driver — check out the paired
fork **as a sibling directory** and build it; its CMake links this repo's
`libopenleap.a` via the default `LEAP_OPEN_RUST_DIR=../OpenLeap/rust-driver`:

```bash
# alongside OpenLeap/ (same parent dir)
git clone https://github.com/julianjc84/monado.git
git -C monado switch leap_open
# build per monado-driver/README.md, then from OpenLeap/:
./run_monado.sh
```

See `ARCHITECTURE.md` for why Mercury is the engine and our Python pipeline is
now the reference/validation oracle.

## Licensing

- This project's code: **BSL-1.0** (see `LICENSE`) — matches Monado, so the
  `leap_open` driver stays upstreamable, and BSL is GPL-compatible so the same
  code can feed GPL-3.0 niri.
- `hand-tracking-models/`: CC-BY-SA 4.0 (Monado/Collabora) — why this stack
  can be shared at all.
- No Ultraleap code, blobs, or SDK output is included or required. The
  protocol knowledge was obtained by observing the device's own USB traffic.

See **`LICENSING.md`** for the full posture: why OpenLeap doesn't breach the
Ultraleap EULA (clean-room USB reverse engineering under interoperability
exceptions; no SDK/Image API used as the frame source), and the lines it must
not cross (never bundle their closed materials; keep GPL linking off-by-
default downstream).

## Status / roadmap

Our own 2D + 3D Python pipeline works and is validated (depth matches ruler;
pinch noise ~1 px after smoothing) — but it is now the **reference/validation
oracle**, not the product. The skeleton engine is Monado's **Mercury**, fed by
our `leap_open` Monado driver. See **`ARCHITECTURE.md`** for the stack decision
(why Mercury over our own pipeline and over the closed Ultraleap stack, what
Mercury outputs, and the open LeapC-shim-vs-native question for niri).

Next: decide the niri integration API (LeapC shim vs. native joint set), tune
Mercury jitter, gesture signal extraction (pinch/grab/swipe with hysteresis)
into niri, and upstream the `leap_open` driver to Monado.
