#!/usr/bin/env python3
"""Phase 0 probe: can we drive the original LMC (dev-kit, FW 1.7.0) via UVC?

Tests, in order:
  1. device discovery   - find the /dev/videoN backed by USB vendor f182
  2. raw capture        - open via V4L2, CONVERT_RGB off, grab frames
  3. de-interleave      - split the YUY2-packed buffer into left/right eyes
  4. LED control        - toggle IR LEDs via the CONTRAST control knock,
                          verify scene brightness responds
  5. exposure control   - via the ZOOM control knock
  6. calibration read   - the sharpness/saturation "property knocking"
                          routine; parse focal length, baseline, distortion
  7. embedded line      - per-frame metadata in the last image row

Artifacts (PNGs, calibration JSON) land in results/.
Run with the Ultraleap tracking service STOPPED.

Protocol reference: leapuvc (https://github.com/leapmotion/leapuvc),
LeapUVC-Manual.pdf. Control mapping (UVC vendor extensions exposed
through standard V4L2/UVC properties):
  CAP_PROP_ZOOM       -> exposure (us)
  CAP_PROP_GAIN       -> analog gain (16..63)
  CAP_PROP_BRIGHTNESS -> digital gain (0..16)
  CAP_PROP_GAMMA      -> gamma enable
  CAP_PROP_CONTRAST   -> selector | (value << 6):
                           0=HDR 1=rotate180 2/3/4=left/center/right LED
  CAP_PROP_SHARPNESS (write addr) + CAP_PROP_SATURATION (read byte)
                      -> calibration memory peek, addresses 100..255
"""

import json
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PASS = []
FAIL = []


def report(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def find_leap_video_device():
    """Scan sysfs for a video4linux node whose USB ancestor has vendor f182."""
    hits = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        dev = node / "device"
        # walk up for idVendor
        p = dev.resolve()
        for _ in range(6):
            vid = p / "idVendor"
            if vid.exists():
                vendor = vid.read_text().strip()
                if vendor == "f182":
                    name = (node / "name").read_text().strip() if (node / "name").exists() else "?"
                    hits.append((int(node.name[5:]), name, (p / "idProduct").read_text().strip()))
                break
            p = p.parent
    return hits


def open_capture(index, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, got


def grab_stereo(cap, resolution, tries=10):
    """Grab one frame and de-interleave into (left, right) uint8 images."""
    w, h = resolution
    for _ in range(tries):
        rval, frame = cap.read()
        if not rval or frame is None:
            continue
        flat = np.asarray(frame, dtype=np.uint8).ravel()
        if flat.size != w * h * 2:
            print(f"    note: unexpected buffer size {flat.size} (want {w*h*2}), shape={frame.shape}")
            continue
        inter = flat.reshape(h, w * 2)
        left = inter[:, 0::2].copy()
        right = inter[:, 1::2].copy()
        return inter, left, right
    return None, None, None


def set_leds(cap, on):
    """LEDs are selectors 2,3,4 on the CONTRAST control."""
    ok = True
    for selector in (2, 3, 4):
        ok &= bool(cap.set(cv2.CAP_PROP_CONTRAST, selector | ((1 if on else 0) << 6)))
        time.sleep(0.01)
    return ok


def mean_brightness(cap, resolution, frames=8, settle=4):
    vals = []
    for i in range(frames + settle):
        _, left, right = grab_stereo(cap, resolution)
        if left is None:
            continue
        if i >= settle:
            vals.append((float(left.mean()) + float(right.mean())) / 2)
    return sum(vals) / len(vals) if vals else None


def read_calibration(cap):
    """Property-knocking: write address to SHARPNESS, read byte from SATURATION."""
    raw = []
    for addr in range(100, 256):
        cap.set(cv2.CAP_PROP_SHARPNESS, addr)
        time.sleep(0.004)
        raw.append(int(cap.get(cv2.CAP_PROP_SATURATION)) & 0xFF)
    fmt = "BBBBIffffffffffffffffffffffffffffffffffffI"
    if struct.calcsize(fmt) != len(raw):
        raise ValueError(f"calcsize {struct.calcsize(fmt)} != {len(raw)}")
    vals = struct.unpack(fmt, bytes(raw))
    cal = {
        "signature": [vals[0], vals[1]],
        "version": vals[2],
        "score": vals[3],
        "timestamp": vals[4],
        "baseline_mm": vals[5] * 1000.0 if vals[5] < 1 else vals[5],
        "baseline_raw": vals[5],
        "q2init": vals[6],
        "checksum": vals[-1],
    }
    off = 7
    for camname in ("left", "right"):
        cal[camname] = {
            "focalLength": vals[off],
            "offset": [vals[off + 1], vals[off + 2]],
            "tangential": [vals[off + 3], vals[off + 4]],
            "radial": list(vals[off + 5 : off + 11]),
            "extr_focalLength": vals[off + 11],
            "extr_center": [vals[off + 12], vals[off + 13]],
            "cayley_rotation": list(vals[off + 14 : off + 17]),
        }
        off += 17
    return cal, raw


def embedded_line(inter, width):
    arr = inter[-1, (width * 2) - 12 :]
    label1 = int(arr[6] >> 4 & 0x1)
    label2 = int(((arr[2] & 0xF) << 4) + (arr[4] & 0xF))
    dark_frame_interval = max(label1, label2) & 0x7F
    exposure = ((int(arr[6] & 0xF) << 5) + int(arr[8] & 0x1F))
    gain = int(arr[10] & 0x1F)
    return dark_frame_interval, exposure, gain


def main():
    print("== Phase 0 probe: LMC dev-kit via UVC ==")

    # service check (warn only - it might just not be claiming the device)
    svc = os.popen("systemctl is-active ultraleap-hand-tracking-service 2>/dev/null").read().strip()
    if svc == "active":
        print("  WARNING: tracking service is ACTIVE - it may own the device. Stop it first.")

    # 1. discovery
    hits = find_leap_video_device()
    if not hits:
        report("device discovery", False, "no /dev/video* with USB vendor f182 - is it plugged in?")
        return finish()
    index, name, product = hits[0]
    report("device discovery", True, f"/dev/video{index} name='{name}' product=0x{product}")

    # 2. capture
    cap, resolution = open_capture(index)
    if cap is None:
        report("open V4L2 capture", False)
        return finish()
    report("open V4L2 capture", True, f"negotiated {resolution[0]}x{resolution[1]}")

    set_leds(cap, True)
    cap.set(cv2.CAP_PROP_ZOOM, 10000)  # exposure in us
    cap.set(cv2.CAP_PROP_GAIN, 30)

    inter, left, right = grab_stereo(cap, resolution, tries=30)
    if left is None:
        report("raw frame capture", False, "no frames with expected buffer size")
        cap.release()
        return finish()
    report("raw frame capture", True, f"buffer {inter.shape}, left/right {left.shape}")
    cv2.imwrite(str(RESULTS / "left.png"), left)
    cv2.imwrite(str(RESULTS / "right.png"), right)
    cv2.imwrite(str(RESULTS / "interleaved.png"), inter)

    # 3. de-interleave sanity: the two eyes see a similar scene -> means are close,
    #    but images differ (parallax) -> not identical
    diff = float(np.abs(left.astype(int) - right.astype(int)).mean())
    report("stereo de-interleave", 0.1 < diff < 200, f"mean |L-R| = {diff:.1f}")

    # 4. LED control: brightness with LEDs on vs off
    b_on = mean_brightness(cap, resolution)
    set_leds(cap, False)
    time.sleep(0.2)
    b_off = mean_brightness(cap, resolution)
    set_leds(cap, True)
    if b_on is None or b_off is None:
        report("LED control", False, "could not sample brightness")
    else:
        report("LED control", b_on > b_off * 1.2 or (b_on - b_off) > 3,
               f"mean brightness LEDs on={b_on:.1f} off={b_off:.1f}")

    # 5. exposure control
    cap.set(cv2.CAP_PROP_ZOOM, 200)
    time.sleep(0.1)
    b_short = mean_brightness(cap, resolution, frames=5)
    cap.set(cv2.CAP_PROP_ZOOM, 30000)
    time.sleep(0.1)
    b_long = mean_brightness(cap, resolution, frames=5)
    cap.set(cv2.CAP_PROP_ZOOM, 10000)
    if b_short is None or b_long is None:
        report("exposure control", False, "could not sample brightness")
    else:
        report("exposure control", b_long > b_short * 1.2 or (b_long - b_short) > 3,
               f"brightness exp=200us {b_short:.1f} vs exp=30000us {b_long:.1f}")

    # 6. calibration
    try:
        cal, raw = read_calibration(cap)
        plausible = (50 < cal["left"]["focalLength"] < 2000
                     and 50 < cal["right"]["focalLength"] < 2000)
        report("calibration read", plausible,
               f"focal L/R = {cal['left']['focalLength']:.1f}/{cal['right']['focalLength']:.1f}, "
               f"baseline_raw = {cal['baseline_raw']:.4f}, version = {cal['version']}")
        (RESULTS / "calibration.json").write_text(json.dumps(cal, indent=2))
        (RESULTS / "calibration_raw.bin").write_bytes(bytes(raw))
    except Exception as e:  # noqa: BLE001 - probe must report, not crash
        report("calibration read", False, repr(e))

    # 7. embedded line + framerate
    t0 = time.time()
    n = 0
    el = None
    while time.time() - t0 < 2.0:
        inter, left, right = grab_stereo(cap, resolution, tries=3)
        if inter is not None:
            n += 1
            el = embedded_line(inter, resolution[0])
    fps = n / 2.0
    report("sustained capture", n > 10, f"{fps:.0f} fps over 2 s")
    if el:
        report("embedded line", True, f"darkFrameInterval={el[0]} exposure={el[1]} gain={el[2]}")

    cap.release()
    finish()


def finish():
    print(f"\n== RESULT: {len(PASS)} pass, {len(FAIL)} fail ==")
    if FAIL:
        print("   failed:", ", ".join(FAIL))
    print(f"   artifacts in {RESULTS}/")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
