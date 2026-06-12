# Phase 2 findings — Mercury skeleton on the LMC's open IR frames

Date: 2026-06-12. Goal: run Monado/Mercury's open hand-tracking ONNX models
on the Leap Motion's IR frames (captured via our open `leap-open` driver)
and judge skeleton quality **before** any niri integration.

## Result: PROVEN but MARGINAL out of the box.

`phase2/skeleton.py` (standalone Python + onnxruntime, no Monado/niri/leapd)
runs Mercury's two models end to end and draws the 21-joint skeleton. On a
real captured frame the detector finds the hand and the keypoint model
produces a skeleton that lands on it — see
`phase2/results/frame15_idx57_left_skel.png` (skeleton fanning over the
palm). So the open models DO generalize to our viewpoint. But confidence is
modest and joints don't snap crisply to the fingers.

## Pipeline (faithful to mercury/hg_model.cpp)

1. **Detection** `grayscale_detection_160x160.onnx`
   - in `inputImg [1,1,160,160]` grayscale, letterboxed, mean-centered to 0.5
   - out `hand_exists/cx/cy/size [1,2]` (2 hand slots; cx,cy in [-1,1])
2. **Keypoint** `grayscale_keypoint_jan18.onnx`
   - in `inputImg [1,1,128,128]` (crop around detection), `lastKeypoints[1,42]=0`,
     `useLastKeypoints[1]=0`
   - out `heatmap_xy [1,21,22,22]` (+ depth, scalars, curls — unused for 2D)
   - per joint: argmax in the 22×22 heatmap + center-of-mass refine → crop px
     → image px.
3. Joint order = `joints_ml_to_xr`: wrist, thumb(mcp/prox/dist/tip),
   then index/middle/ring/little (prox/int/dist/tip each). 21 joints.

## What works / what doesn't

- ✅ **End-to-end plumbing**: our open IR frames → Mercury models → skeleton.
- ✅ **Detection fires** on a real hand (peak `hand_exists` 0.64).
- ⚠️ **Orientation is required**: raw frames give ~0.00; **`--rot 270`** gives
  0.64. Our device is "fixed-inverted" and Mercury expects hands roughly
  upright. (rot270 was best across a rot/flip sweep.)
- ⚠️ **Keypoint confidence low** (`kp_conf` ~0.12–0.28) and joints are
  loosely placed.

### Why quality is marginal (in order of fixability)

1. **Framing**: in the test the hand drifted to the frame edge, so the
   keypoint crop got a partial hand. Centering the hand should help a lot.
2. **Exposure**: the hand was dim/far — AE hit its 8000 µs ceiling at mean
   ~28 (target 70). A closer hand + capping exposure lower (less motion
   blur) gives the models more signal. (First-ever capture, hand close, hit
   mean 100 — much brighter.)
3. **Domain gap (fundamental)**: Mercury trained on **head-mounted views of
   the back of the hand at arm's length**; ours is a **close, palm-facing,
   desk-mounted** view. This is the real ceiling on out-of-the-box quality.

## Assessment for niri integration

Out-of-the-box Mercury is **not yet crisp enough** to replace Gemini's
`is_extended`/pinch/grab reliability for niri gestures — the skeleton is
roughly right but jittery and low-confidence on the palm-facing viewpoint.
Two paths to "good enough":

- **Cheap wins first**: better capture conditioning — center + brighten the
  hand, lock orientation to rot270, cap exposure, temporal smoothing
  (`lastKeypoints`/`useLastKeypoints` which we currently zero out — Mercury
  uses them for stability), and stereo fusion (we only used the left eye).
- **Real fix if needed**: fine-tune the keypoint model on our viewpoint
  using `../mercury_train` — record palm-facing IR hands and retrain. This
  is the documented purpose of that repo and the path to closing the domain
  gap. (Must avoid using the closed Gemini SDK to generate labels — EULA
  §3.1.5; use mercury_train's own annotation/synthetic data.)

## Tooling added

- `phase2/skeleton.py` — standalone detection+keypoint+draw, `--dir`/`--rot`/
  `--flip`/`--gain`/`--crop-scale`. Runs on `leap-open/captured_frames/*.pgm`.
- `phase2/results/` — annotated skeleton overlays.
- venv now has `onnxruntime`.

## Next steps

1. **Conditioned re-test**: capture with the hand centered and close
   (brighter), exposure capped ~1500 µs, then re-run — to see the realistic
   best-case 2D quality.
2. **Temporal + stereo**: feed `lastKeypoints` across frames; run both eyes
   and triangulate for 3D (needed for `LeapHandData`).
3. Decide: condition-and-smooth vs fine-tune. Only after 2D quality is
   convincing do we wire a `LeapHandData` shim into niri.

## UPDATE (2026-06-12): undistortion step built

The LMC stores per-camera calibration on-device; we now read it and remove
the fisheye before inference (Mercury normally undistorts upstream — we'd
been feeding raw fisheye).

- **`leap-open calib`** — reads the 156-byte calibration via the LMC's
  "property-knocking" backdoor (write address to Processing Unit sharpness
  selector 0x08, read byte from saturation 0x07; PU = unit 5, confirmed
  from the VC descriptors). Dumps raw bytes to stdout.
  Confirmed valid: signature "CA", **baseline 40.01 mm** (matches the LMC's
  known stereo separation), focal ~132 px/eye.
- **`phase2/calibration.py`** — parses those bytes (leapuvc layout), inverts
  Leap's radial model into OpenCV distCoeffs via curve fit, builds per-eye
  `initUndistortRectifyMap` maps. `build`/`show` subcommands; saved to
  `phase2/calib.npz`. Stored bytes also kept in `phase2/calibration.bin`.
- **`live_viewer.py`** — loads `calib.npz`, applies `cv2.remap` to the
  selected eye (in native frame, before flip/rot) with a live **undistort**
  toggle (default on). Compare on/off in real time.

Note: this is OpenCV rectilinear undistortion. Mercury's models may have
trained on a *stereographic* unprojection (`hg_stereographic_unprojection.hpp`)
rather than pinhole-rectilinear; if undistort helps but isn't enough,
matching that exact projection is the next refinement.

## UPDATE (2026-06-12): camera "glitching" solved — host-side frame tearing

The visible glitch-every-few-seconds (and an invisible >50% frame loss) was
NOT the device, power, or the dark-frame strobe — diag counters showed
`short/128 = 130–184` with **zero** USB errors/timeouts. Cause: leap-open's
single-threaded loop blocked writing 307 KB frames into the viewer's ~64 KB
pipe; while blocked, the bulk endpoint went unserviced and the device FIFO
overflowed → torn frames.

Fix (in `leap-open stream`):
1. **Decoupled drain/writer**: a dedicated thread does only USB-read +
   UVC reassembly into a 1-slot mailbox; the writer ships the *latest* frame
   to stdout and drops whole frames if the viewer is slow (never blocks the
   drain).
2. **FID-aware framing**: a lost EOF payload no longer merges two frames —
   the frame-ID toggle bit forces the boundary.
3. **SCHED_FIFO 10 on the drain thread** (root via pkexec): a payload lands
   every ~470 µs; normal-priority scheduling lost ~15% of frames idle and
   ~50%+ when onnx ran. RT priority preempts the inference load instantly.
4. **onnxruntime capped to 2 threads** in the viewer (it grabbed every core
   and starved the drain).

Result (user-confirmed): `short/128` = 0–2 (~1% worst case), `drop/128` = 0,
no visible glitching. The `diag:` stderr line (every 128 frames) now reports
dark/short/forced/drop/timeouts/errors deltas for future regressions. The
viewer gained a **skeleton on/off toggle** (default off = raw-inspection
mode) and frame-integrity readouts.

Open issue: AE is pinned at the 8000 µs ceiling with the center-only LED
(mean ~30 vs target 70) → underexposed + motion-blur-prone. Next: wire
`leds`/`exp`/`ae` stdin controls into the viewer UI and test 2–3 LEDs
(errors=0 throughout shows the bus handles it; the earlier "brownout" fear
was actually this host-side tearing).
