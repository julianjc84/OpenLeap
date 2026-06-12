# `leap_open` — Monado driver for the original Leap Motion Controller

> **This directory is design/reference docs — nothing here is compiled.** The
> built C driver lives in the monado fork at
> `monado/src/xrt/drivers/leap_open/`; the Rust capture half it links lives in
> `../rust-driver/` (built as `libopenleap.a`). See "Build milestones" below
> for how the two halves meet.

Goal: feed the Leap's stereo IR into Monado's **Mercury** hand tracker as a
first-class camera driver, so we get the full 26-joint 3D hand skeleton
(occlusion inference, temporal stability, hand-size estimation) **without
forking Mercury** — and contribute the driver upstream.

This is the *open* counterpart to Monado's existing `ultraleap_v2`/`ultraleap_v5`
drivers, which depend on the dead closed Ultraleap tracking service. There is
currently **no open camera driver for the original Leap in Monado** — this fills
that gap.

## Architecture (how it plugs in)

```
 our Rust driver            this C driver (leap_open)             Monado
 (rust-driver/)             src/xrt/drivers/leap_open/
 ───────────────            ─────────────────────────            ──────
 bringup (358 writes)  ──▶  leap_open_open()        \
 SCHED_FIFO drain      ──▶  leap_open_next_frame()   } via C ABI  ──▶ xrt_frame ──▶ ht driver
 exposure/AE/LEDs      ──▶  leap_open_set_*()        /                (Mercury) ──▶ xrt_hand_joint_set
 156-byte calib        ──▶  leap_open_read_calib()  /                            ──▶ XR_EXT_hand_tracking
```

Nothing in `rust-driver/` is reimplemented in C — the C driver is a thin shim
that calls our Rust logic through a `#[no_mangle] extern "C"` ABI (built as a
`staticlib`). The C side only does Monado plumbing: `xrt_fs` frameserver,
`u_var` controls, `u_sink_debug` preview, and the `t_stereo_camera_calibration`
hand-off to the `ht` driver.

## What maps where (answers "do we keep our controls?")

| Our control            | In `leap_open`                         | Live-tunable? |
| ---------------------- | -------------------------------------- | ------------- |
| Exposure µs            | `u_var_draggable_u16` → `set_exposure` | ✅ monado-gui |
| Auto-exposure          | `u_var` bool → `set_ae`                | ✅ monado-gui |
| IR LED mask (L\|C\|R)  | `u_var` u8 → `set_leds`                | ✅ monado-gui |
| **See the IR Mercury sees** | `u_sink_debug` "Left/Right IR" tabs | ✅ live preview |
| Rotation / flip H/V    | fixed orientation matched to calib + mounting pose | n/a (one correct orientation) |
| Undistort / crop       | **Mercury does it** (`hg_image_distorter`, `hg_sync`) | removed |
| Eye select / gain      | debug-view only (Mercury needs both eyes; gain was screen-only) | n/a |

## Calibration (the one device-unique seam)

`tracking/calibration.py` already extracts everything Monado's
`t_camera_calibration` needs, per eye:

- `K` (3×3 intrinsics) → `intrinsics[3][3]`
- 8 OpenCV distortion coeffs → `T_DISTORTION_OPENCV_RADTAN_8`
- Cayley `rotation` (→ stereo `camera_rotation[3][3]`)
- `baseline_mm` (→ `camera_translation[3]`, mm vs m TBD against Mercury)

Plan: a small exporter writes the Leap calibration in Monado's calibration
format so the driver loads it via `t_stereo_camera_calibration_from_json_v2()`
(or fills the struct directly). RADTAN_8 because we already fit k1,k2,p1,p2,
k3,k4,k5,k6.

## Build milestones

0. **Scaffold + plan** (this directory) — done, device-independent.
1. **Tracking-only Monado build** — ✅ DONE / validated. *No package installs
   needed* — eigen, opencv 4.13, onnxruntime-cpu, sdl2-compat, hidapi,
   vulkan-icd-loader are already present (`find_package` locates them; the
   pkg-config names just differ). The slim config below compiles the whole
   hand-tracking stack (Mercury + kine_lm + `ht` driver) in ~25 s with the
   compositor / OpenXR / service / IPC / all VR drivers OFF:

   ```
   cmake -S <monado> -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DXRT_MODULE_COMPOSITOR=OFF -DXRT_MODULE_COMPOSITOR_MAIN=OFF -DXRT_MODULE_COMPOSITOR_NULL=OFF \
     -DXRT_MODULE_IPC=OFF -DXRT_MODULE_OPENXR_STATE_TRACKER=OFF -DXRT_MODULE_OPENVR_STATE_TRACKER=OFF \
     -DXRT_FEATURE_OPENXR=OFF -DXRT_FEATURE_SERVICE=OFF -DXRT_FEATURE_STEAMVR_PLUGIN=OFF \
     -DXRT_FEATURE_SLAM=OFF -DXRT_FEATURE_WINDOW_PEEK=OFF -DXRT_FEATURE_RENDERDOC=OFF \
     -DXRT_MODULE_MONADO_CLI=OFF -DXRT_MODULE_MONADO_GUI=ON \
     -DXRT_BUILD_SAMPLES=OFF -DXRT_BUILD_DRIVER_HANDTRACKING=ON \
     -DXRT_BUILD_DRIVER_{ARDUINO,…all other VR drivers…}=OFF
   # ninja drv_ht  ->  libdrv_ht.a + libt_ht_mercury*.a   (25 s, zero installs)
   ```
   Built libs: `libt_ht_mercury{,_model,_distorter}.a`, `libt_ht_mercury_kine_lm.a`,
   `libdrv_ht.a`, `libhand_async.a`, `libaux_tracking.a`, `libaux_onnx.a`.
2. **Wire the driver** — ✅ DONE. Lives in the fork
   (`CLAUDE_PROJECTS/monado`, branch `leap_open`):
   - `src/xrt/drivers/leap_open/` — the `xrt_fs` driver (brings the Leap up via
     `libopenleap.a`, de-interleaves to two L8 frames, loads the calibration_v2
     JSON, `u_var` controls + `u_sink_debug` preview). CMake builds
     `libopenleap.a` via cargo (`LEAP_OPEN_RUST_DIR`) and links it.
   - `src/xrt/targets/leap_open_ht/` — standalone harness: `leap_open` → genlock
     → `ht_device_create` (Mercury) → prints 3D joints. No monado-service.
   - `rust-driver` is lib+bin with the `leap_open_*` C ABI staticlib (in OpenLeap).
   - `calibration.py monado` produces `calib_monado.json` (this dir's sibling).

   Build + (headless) run:
   ```bash
   cmake -B build -DXRT_BUILD_DRIVER_LEAP_OPEN=ON   # + the milestone-1 slim flags
   ninja -C build leap_open_ht
   LEAP_OPEN_SEQ=/path/to/bringup_seq.txt LEAP_OPEN_CALIB=/path/to/calib_monado.json \
     ./build/src/xrt/targets/leap_open_ht/leap_open_ht
   ```
   The whole chain (Rust ABI + driver + calibration loader + Mercury) compiles,
   links, and reaches device bring-up at runtime — verified with no device.
   (`leap_open_driver.c` in THIS dir was the design draft; the real one is in the fork.)
3. **See & test** — connect the device; run the harness (prints live 3D joints)
   and/or the debug GUI (`XRT_FEATURE_DEBUG_GUI`, SDL2) for the "Left/Right IR"
   preview, exposure/AE/LED `u_var` sliders, and Mercury's hand overlay. Tune,
   then upstream the driver.

## The C ABI (added to `rust-driver`)

```rust
// rust-driver/src/ffi.rs  (built as staticlib at milestone 2)
#[no_mangle] pub extern "C" fn leap_open_open() -> *mut LeapOpen;
#[no_mangle] pub extern "C" fn leap_open_read_calib(h: *mut LeapOpen, out: *mut u8 /*156*/) -> i32;
#[no_mangle] pub extern "C" fn leap_open_start(h: *mut LeapOpen) -> i32;
#[no_mangle] pub extern "C" fn leap_open_next_frame(h: *mut LeapOpen, buf: *mut u8, len: usize) -> i32; // blocks
#[no_mangle] pub extern "C" fn leap_open_set_exposure(h: *mut LeapOpen, us: u32);
#[no_mangle] pub extern "C" fn leap_open_set_ae(h: *mut LeapOpen, on: bool);
#[no_mangle] pub extern "C" fn leap_open_set_leds(h: *mut LeapOpen, mask: u8);
#[no_mangle] pub extern "C" fn leap_open_close(h: *mut LeapOpen);
```

These wrap the existing `rust-driver` logic (bringup, drain thread + mailbox,
control writes, calib read) — no reimplementation, just a relocation of
`main.rs` into a `lib.rs` + this thin module, with `main.rs` reduced to a shim.

The C side lives in the monado fork (`CLAUDE_PROJECTS/monado`, branch
`leap_open`) at `src/xrt/drivers/leap_open/leap_open_driver.c` — modeled on
Monado's `depthai` driver. That is the **single source of truth** for the C
driver and the only copy that gets built; this directory is design/reference
docs, not compiled.
