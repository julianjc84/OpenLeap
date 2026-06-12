#!/usr/bin/env python3
"""Parse the LMC's on-device calibration (156 bytes from `openleap calib`)
and build per-eye OpenCV undistortion maps, so we can remove the fisheye
distortion before feeding frames to the hand-tracking models.

Ported from leapuvc.py's retrieveLeapCalibration. The device stores, per
camera: focal length, principal-point offset, tangential + radial distortion
(in Leap's own inverse form), plus stereo extrinsics. We invert the radial
model into OpenCV distCoeffs via curve fit, then initUndistortRectifyMap.

Usage:
  calibration.py build <calibration.bin> [--res 640x240] [-o calib.npz]
  calibration.py show  <calib.npz>          # print parsed values
"""

import argparse
import json
import struct
import sys

import cv2
import numpy as np
from scipy.optimize import curve_fit, least_squares

# leapuvc layout: 4 bytes (sig,sig,ver,score), u32 timestamp, then 36 floats:
# baseline, Q2Init, and 17 per camera (focal, offset^2, tangential^2, radial^6,
# deprecated-focal, center^2, CayleyRotation^3), then a u32 checksum = 156 B.
STRUCT_FMT = "BBBBI" + "f" * 36 + "I"


def _normal_warp(r, k1, k2, k3, k4, k5, k6):
    return (1.0 + r * (k1 + r * (k2 + r * k3))) / (1.0 + r * (k4 + r * (k5 + r * k6)))


def _mono1(r, k1, k2, k3, k4, k5, k6):
    return r * (_normal_warp(r, k1, k2, k3, k4, k5, k6) ** 3)


def _mono2(r, k1, k2, k3, k4, k5, k6):
    return r * (_normal_warp(r, k1, k2, k3, k4, k5, k6) ** 2)


def parse(raw, resolution=(640, 240), rectify=False):
    """raw: 156 bytes. resolution: (w,h) of one eye. Returns dict per camera.
    rectify=False keeps each eye in its own frame (mono undistort only);
    rectify=True applies the stereo rectification rotation AND projects both
    eyes through a COMMON camera matrix, so a scene point lands on the same
    row in both eyes and disparity is purely horizontal — required for
    triangulation. meta["K_common"] holds that shared matrix.
    """
    if len(raw) < struct.calcsize(STRUCT_FMT):
        raise ValueError(f"need {struct.calcsize(STRUCT_FMT)} bytes, got {len(raw)}")
    a = struct.unpack(STRUCT_FMT, raw[: struct.calcsize(STRUCT_FMT)])
    meta = {"sig": (a[0], a[1]), "version": a[2], "score": a[3],
            "timestamp": a[4], "baseline_mm": a[5]}
    cams = {}
    off = 7  # first per-camera float index
    w, h = resolution
    aspect = h / 480.0
    for name in ("left", "right"):
        focal = a[off]
        offset = (a[off + 1], a[off + 2])
        tangential = (a[off + 3], a[off + 4])
        radial = list(a[off + 5:off + 11])
        rotation = list(a[off + 14:off + 17])
        off += 17

        K = np.array([[focal, 0.0, 320 + offset[0]],
                      [0.0, focal * aspect, (240 + offset[1]) * aspect],
                      [0.0, 0.0, 1.0]], np.float32)

        # invert Leap's radial model into OpenCV distCoeffs (leapuvc method)
        xdata = np.linspace(-0.99, -0.35, 33)
        xdata = (1.0 / (xdata ** 2)) - 1.0
        try:
            ydata = _mono1(xdata, *radial)
            k, _ = curve_fit(_mono1, ydata, xdata, maxfev=10000)
        except Exception:
            ydata = _mono2(xdata, *radial)
            k, _ = curve_fit(_mono2, ydata, xdata, maxfev=10000)
        dist = np.array([k[0], k[1], tangential[0], tangential[1],
                         k[2], k[3], k[4], k[5]], np.float32)
        cams[name] = {"focal": focal, "offset": offset, "K": K, "dist": dist,
                      "rotation": rotation}

    K_common = ((cams["left"]["K"] + cams["right"]["K"]) * 0.5
                if rectify else None)
    meta["K_common"] = K_common
    for name in ("left", "right"):
        c = cams[name]
        R = _cayley(-np.asarray(c["rotation"])) if rectify else None
        Knew = K_common if rectify else c["K"]
        m1, m2 = cv2.initUndistortRectifyMap(c["K"], c["dist"], R, Knew,
                                             (w, h), cv2.CV_16SC2)
        c["map1"], c["map2"] = m1, m2
    return meta, cams


def _cayley(p):
    x, y, z = p[0] * 2, p[1] * 2, p[2] * 2
    xx, yy, zz = p[0] * p[0], p[1] * p[1], p[2] * p[2]
    xy, yz, zx = p[0] * p[1] * 2, p[1] * p[2] * 2, p[2] * p[0] * 2
    m = np.array([[1 + xx - yy - zz, xy - z, zx + y],
                  [xy + z, 1 - xx + yy - zz, yz - x],
                  [zx - y, yz + x, 1 - xx - yy + zz]], np.float32)
    return m / (1 + xx + yy + zz)


def undistort(eye_img, cam):
    return cv2.remap(eye_img, cam["map1"], cam["map2"], cv2.INTER_LINEAR)


def save_npz(path, meta, cams, resolution):
    extra = {}
    if meta.get("K_common") is not None:  # rectified build: shared intrinsics
        Kc = meta["K_common"]
        extra = {"fx": Kc[0, 0], "fy": Kc[1, 1], "cx": Kc[0, 2], "cy": Kc[1, 2],
                 "rectified": True}
    np.savez(path,
             res=np.array(resolution),
             baseline_mm=meta["baseline_mm"],
             left_map1=cams["left"]["map1"], left_map2=cams["left"]["map2"],
             right_map1=cams["right"]["map1"], right_map2=cams["right"]["map2"],
             left_K=cams["left"]["K"], right_K=cams["right"]["K"], **extra)


def load_npz(path):
    d = np.load(path)
    out = {
        "res": tuple(int(x) for x in d["res"]),
        "baseline_mm": float(d["baseline_mm"]),
        "left": {"map1": d["left_map1"], "map2": d["left_map2"], "K": d["left_K"]},
        "right": {"map1": d["right_map1"], "map2": d["right_map2"], "K": d["right_K"]},
        "rectified": bool(d["rectified"]) if "rectified" in d else False,
    }
    if out["rectified"]:
        for k in ("fx", "fy", "cx", "cy"):
            out[k] = float(d[k])
    return out


def _refit_radtan5(K, dist8, w, h):
    """Refit OpenCV's 5-param radtan (k1,k2,p1,p2,k3) to our validated 8-param
    rational model over the image FOV. Monado's calibration JSON loader only
    accepts radtan5 / kb4; radtan5 is sub-0.25 px in the central FOV where hands
    sit (full RADTAN_8 fidelity, ~rational, is a later upgrade via direct
    struct fill). Returns ([k1,k2,p1,p2,k3], max_px_err)."""
    K = np.asarray(K, np.float64)
    d8 = np.asarray(dist8, np.float64)
    xs = np.linspace(-w * 0.5, w * 0.5, 48)
    ys = np.linspace(-h * 0.5, h * 0.5, 20)
    gx, gy = np.meshgrid(xs, ys)
    rays = np.stack([gx.ravel() / K[0, 0], gy.ravel() / K[1, 1], np.ones(gx.size)], 1)
    rvec = tvec = np.zeros(3)
    truth = cv2.projectPoints(rays, rvec, tvec, K, d8)[0].reshape(-1, 2)

    def resid(p):
        pr = cv2.projectPoints(rays, rvec, tvec, K, p)[0].reshape(-1, 2)
        return (pr - truth).ravel()

    r = least_squares(resid, d8[:5].copy(), method="lm")
    return r.x, float(np.abs(r.fun).max())


def stereo_extrinsics(cams, baseline_mm):
    """Relative pose left-camera -> right-camera from the per-camera Cayley
    rotations and the baseline. Returns (R 3x3, T 3-vec in METERS, E, F). T is
    in metres so Mercury emits joints in the OpenXR metre convention. Validated:
    cv2.stereoRectify recovers P2 tx/fx == -baseline."""
    rl = _cayley(-np.asarray(cams["left"]["rotation"], float))
    rr = _cayley(-np.asarray(cams["right"]["rotation"], float))
    r = rr.T @ rl  # rotate a point from the left cam frame into the right cam frame
    base_m = baseline_mm / 1000.0
    c_right_in_left = rl.T @ np.array([base_m, 0.0, 0.0])
    t = (-r @ c_right_in_left).reshape(3)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    e = tx @ r
    kl = np.asarray(cams["left"]["K"], float)
    kr = np.asarray(cams["right"]["K"], float)
    f = np.linalg.inv(kr).T @ e @ np.linalg.inv(kl)
    return r, t, e, f


def export_monado(meta, cams, resolution, path):
    """Write the Leap's on-device calibration as a Monado calibration_v2 JSON
    (pinhole_radtan5 per eye + opencv_stereo_calibrate), loadable by the
    leap_open driver via t_stereo_camera_calibration_from_json_v2."""
    w, h = resolution
    r, t, e, f = stereo_extrinsics(cams, meta["baseline_mm"])
    errs = {}

    def cam(name):
        k = np.asarray(cams[name]["K"], float)
        d5, err = _refit_radtan5(k, cams[name]["dist"], w, h)
        errs[name] = round(err, 3)
        return {
            "name": f"Leap {name}",
            "model": "pinhole_radtan5",
            "intrinsics": {"fx": float(k[0, 0]), "fy": float(k[1, 1]),
                           "cx": float(k[0, 2]), "cy": float(k[1, 2])},
            "distortion": {"k1": float(d5[0]), "k2": float(d5[1]), "p1": float(d5[2]),
                           "p2": float(d5[3]), "k3": float(d5[4])},
            "resolution": {"width": int(w), "height": int(h)},
        }

    cameras = [cam("left"), cam("right")]
    doc = {
        "$schema": "https://monado.pages.freedesktop.org/monado/calibration_v2.schema.json",
        "metadata": {"version": 2,
                     "source": "OpenLeap calibration.py (Leap on-device, radtan5 refit)",
                     "radtan5_max_px_err": errs, "baseline_mm": round(meta["baseline_mm"], 4)},
        "cameras": cameras,
        "opencv_stereo_calibrate": {
            "rotation": [float(x) for x in r.ravel()],
            "translation": [float(x) for x in t.ravel()],
            "essential": [float(x) for x in e.ravel()],
            "fundamental": [float(x) for x in f.ravel()],
        },
    }
    with open(path, "w") as fp:
        json.dump(doc, fp, indent=2)
    return doc, errs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("bin")
    b.add_argument("--res", default="640x240")
    b.add_argument("-o", default="calib.npz")
    b.add_argument("--no-rectify", action="store_true",
                   help="mono undistort only (no stereo rectification)")
    s = sub.add_parser("show")
    s.add_argument("npz")
    m = sub.add_parser("monado", help="export Monado calibration_v2 JSON for the leap_open driver")
    m.add_argument("bin")
    m.add_argument("--res", default="640x240")
    m.add_argument("-o", default="calib_monado.json")
    args = ap.parse_args()

    if args.cmd == "build":
        w, h = (int(x) for x in args.res.split("x"))
        raw = open(args.bin, "rb").read()
        meta, cams = parse(raw, (w, h), rectify=not args.no_rectify)
        save_npz(args.o, meta, cams, (w, h))
        print(f"calibration: sig={meta['sig']} ver={meta['version']} "
              f"baseline={meta['baseline_mm']:.2f}mm")
        for n in ("left", "right"):
            print(f"  {n}: focal={cams[n]['focal']:.1f} "
                  f"offset={cams[n]['offset']} dist={cams[n]['dist'][:4].round(3)}")
        print(f"wrote undistortion maps -> {args.o}  (res {w}x{h})")
    elif args.cmd == "show":
        c = load_npz(args.npz)
        print(f"res={c['res']} baseline={c['baseline_mm']:.2f}mm")
        print("left K=\n", c["left"]["K"])
    elif args.cmd == "monado":
        w, h = (int(x) for x in args.res.split("x"))
        raw = open(args.bin, "rb").read()
        meta, cams = parse(raw, (w, h), rectify=True)
        _, errs = export_monado(meta, cams, (w, h), args.o)
        print(f"wrote Monado calibration_v2 -> {args.o}")
        print(f"  baseline={meta['baseline_mm']:.3f}mm  radtan5 max px err: "
              f"L={errs['left']} R={errs['right']}")


if __name__ == "__main__":
    main()
