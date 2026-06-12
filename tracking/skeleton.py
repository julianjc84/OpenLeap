#!/usr/bin/env python3
"""Standalone 2D hand-skeleton test: run Monado/Mercury's open ONNX models on
the Leap Motion's IR frames and draw the 21-joint skeleton — no Monado, no
niri, no leapd. Purpose: judge whether the open models track our desk-mounted,
palm-up IR viewpoint well enough before investing in niri integration.

Pipeline (faithful to monado/.../mercury/hg_model.cpp):
  1. detection net  grayscale_detection_160x160.onnx
       in:  inputImg [1,1,160,160] grayscale, letterboxed, mean-centered to 0.5
       out: hand_exists[1,2], cx[1,2], cy[1,2], size[1,2]   (cx,cy in [-1,1])
  2. crop around (cx,cy,size), resize to 128, run keypoint net
       grayscale_keypoint_jan18.onnx
       in:  inputImg [1,1,128,128], lastKeypoints[1,42]=0, useLastKeypoints[1]=0
       out: heatmap_xy [1,21,22,22]  (+ depth, scalars, curls — unused here)
  3. per joint: argmax in the 22x22 heatmap, refine by center-of-mass,
     map crop->image, draw.

Usage:
  skeleton.py <image.pgm|png> [--rot 0|90|180|270] [--flip] [--out annotated.png]
  skeleton.py --dir captured_frames           # process all *_left.pgm
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

MODELS = os.path.join(os.path.dirname(__file__), "..", "hand-tracking-models")
DET_SIZE = 160
KP_IN = 128
HM = 22  # heatmap spatial size

# 21-joint order from hg_model.cpp joints_ml_to_xr
JOINTS = [
    "wrist",
    "thumb_mcp", "thumb_prox", "thumb_dist", "thumb_tip",
    "index_prox", "index_int", "index_dist", "index_tip",
    "middle_prox", "middle_int", "middle_dist", "middle_tip",
    "ring_prox", "ring_int", "ring_dist", "ring_tip",
    "little_prox", "little_int", "little_dist", "little_tip",
]
# bones: wrist -> each finger base, then the finger chain
BONES = []
for base in (1, 5, 9, 13, 17):
    BONES.append((0, base))
    BONES += [(base + k, base + k + 1) for k in range(3)]
FINGER_COLORS = {  # BGR per finger for readability
    "thumb": (0, 0, 255), "index": (0, 165, 255), "middle": (0, 255, 255),
    "ring": (0, 255, 0), "little": (255, 128, 0), "wrist": (255, 255, 255),
}


def finger_of(idx):
    if idx == 0:
        return "wrist"
    return ["thumb", "index", "middle", "ring", "little"][(idx - 1) // 4]


def normalize_gray(img_u8):
    """Mercury's normalizeGrayscaleImage: /255 then shift so mean == 0.5."""
    f = img_u8.astype(np.float32) / 255.0
    f += 0.5 - f.mean()
    return f


def letterbox(img, size):
    """Resize preserving aspect into a square `size`, centered, zero-padded.
    Returns (canvas, scale, off_x, off_y) to map detections back."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, ox, oy


def argmax_refine(hm):
    """Coarse argmax + center-of-mass refinement in a +-k window (hg_model.cpp)."""
    cy, cx = np.unravel_index(int(np.argmax(hm)), hm.shape)
    k = 4
    y0, y1 = max(0, cy - k), min(hm.shape[0], cy + k + 1)
    x0, x1 = max(0, cx - k), min(hm.shape[1], cx + k + 1)
    win = hm[y0:y1, x0:x1]
    s = win.sum()
    if s <= 0:
        return float(cx), float(cy), float(hm.max())
    ys, xs = np.mgrid[y0:y1, x0:x1]
    rx = (win * (xs + 0.5)).sum() / s
    ry = (win * (ys + 0.5)).sum() / s
    return float(rx), float(ry), float(hm.max())


# hand_exists thresholds: below HAND_MIN a detection is treated as noise and
# nothing is drawn (empty scenes always have *some* argmax blob — arm, ceiling,
# AE-amplified noise — which scored a phantom skeleton when we drew it
# unconditionally). Real hands score ~0.6-0.7 in good light; the second slot
# is held to a stricter bar since it fires on duplicates/noise more often.
HAND_MIN = 0.40
SECOND_HAND_MIN = 0.50
# a coasting track (detector lost it, keypoints still good) survives while its
# mean keypoint confidence stays above this
COAST_MIN = 0.32


def _track_hand(kp, gray, ex, ey, sz, scale, args, prev_pts=None, flip=False):
    """Crop around one detection (eye coords) and run the keypoint net.
    prev_pts: last frame's 21 joints in eye coords (or None). Mercury feeds
    these back as `lastKeypoints` — the previous joints mapped into the
    *current* crop, normalized to (-1,1), with useLastKeypoints=1 — and the
    net uses them as a temporal prior (hg_model.cpp). Big stability win vs
    solving every frame from scratch.
    flip: mirror the crop before inference and un-mirror the results. The
    keypoint net is trained on ONE handedness (Mercury mirrors the other
    hand's crop — make_projection_instructions' flip arg); without this the
    thumb lands on the pinky side of a wrong-handed hand.
    Returns (pts, confs, box) with joints in eye coords."""
    crop_half = max(8.0, (sz / scale) * 0.5 * args.crop_scale)
    x0 = int(round(ex - crop_half)); y0 = int(round(ey - crop_half))
    x1 = int(round(ex + crop_half)); y1 = int(round(ey + crop_half))
    H, W = gray.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    crop = np.zeros((y1 - y0, x1 - x0), np.uint8)
    crop[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = gray[cy0:cy1, cx0:cx1]
    crop128 = cv2.resize(crop, (KP_IN, KP_IN), interpolation=cv2.INTER_AREA)
    if flip:
        crop128 = cv2.flip(crop128, 1)
    kin = normalize_gray(crop128)[None, None]
    last = np.zeros((1, 42), np.float32)
    use = np.zeros((1,), np.float32)
    if prev_pts is not None:
        cw = x1 - x0
        for j, (jx, jy) in enumerate(prev_pts):
            xn = ((jx - x0) / cw) * 2.0 - 1.0
            last[0, j * 2 + 0] = -xn if flip else xn
            last[0, j * 2 + 1] = ((jy - y0) / cw) * 2.0 - 1.0
        use[0] = 1.0
    hmaps = kp.run(["heatmap_xy"], {
        "inputImg": kin, "lastKeypoints": last, "useLastKeypoints": use
    })[0][0]  # [21,22,22]

    pts = []
    confs = []
    cw = (x1 - x0)
    for j in range(21):
        hx, hy, conf = argmax_refine(hmaps[j])
        if flip:
            hx = HM - hx  # mirror back into the unflipped crop
        jx = x0 + (hx / HM) * cw
        jy = y0 + (hy / HM) * cw
        pts.append((jx, jy))
        confs.append(conf)
    return pts, confs, (x0, y0, x1, y1)


def triangulate(pts_l, pts_r, confs_l, confs_r, cal,
                min_conf=0.15, max_row_err=8.0, z_range=(60.0, 700.0)):
    """Rectified stereo triangulation of the 21 joints.

    pts_l/pts_r: [21][2] joint px in the LEFT/RIGHT rectified eye images.
    cal: calibration dict from calibration.load_npz (must be a rectified
    build: fx/fy/cx/cy shared intrinsics + baseline_mm).

    Per joint: disparity d = x_l - x_r, depth Z = fx*B/d, then X,Y by
    unprojecting the left-eye pixel. Gates: both eyes confident, rows agree
    (epipolar check), depth plausible. Gated-out joints get the hand's median
    depth (a hand is shallow relative to its distance, so this is a decent
    fill) and are flagged invalid.

    Returns (xyz [21,3] float mm in the left-camera frame, valid [21] bool).
    """
    pl = np.asarray(pts_l, np.float32)
    pr = np.asarray(pts_r, np.float32)
    cl = np.asarray(confs_l, np.float32)
    cr = np.asarray(confs_r, np.float32)
    fx, fy, cx, cy = cal["fx"], cal["fy"], cal["cx"], cal["cy"]
    base = cal["baseline_mm"]

    disp = pl[:, 0] - pr[:, 0]
    # eye labeling can be swapped relative to physical left/right; the sign
    # of the disparity tells us. Depth magnitudes are unaffected.
    if np.median(disp) < 0:
        disp = -disp
    row_err = np.abs(pl[:, 1] - pr[:, 1])
    valid = (cl > min_conf) & (cr > min_conf) & (disp > 1.0) \
        & (row_err < max_row_err)
    z = np.full(21, np.nan, np.float32)
    z[valid] = fx * base / disp[valid]
    valid &= (z > z_range[0]) & (z < z_range[1])
    z[~valid] = np.nan
    if np.any(valid):
        z = np.where(np.isnan(z), np.nanmedian(z), z)
    else:
        return np.full((21, 3), np.nan, np.float32), valid
    x = (pl[:, 0] - cx) * z / fx
    y = (pl[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=1), valid


def detect_arm(gray, box, pts):
    """Find the forearm feeding a hand and root it on the net's wrist.

    The wrist (Mercury joint 0, pts[0]) is the shared point between hand and
    forearm on a real arm, so we anchor the forearm there rather than letting
    it guess its own wrist — they come out as one rigid chain, not two
    free-floating estimates. No ML — threshold + connected components; the
    forearm direction is taken from where the bright blob extends AWAY from
    that wrist. Returns {dir: unit 2-vec arm->hand, wrist2d: (x,y) = the net
    wrist, conf: 0..1} or None.

    Why: the kinematic fit's weakest-observed DOF is wrist orientation, and
    Mercury's wrist joint has the lowest keypoint confidence. The forearm is
    the biggest, most stable structure in frame and anatomically upstream of
    everything — it anchors the wrist's orientation.
    """
    H, W = gray.shape
    x0, y0, x1, y1 = (max(0, box[0]), max(0, box[1]),
                      min(W, box[2]), min(H, box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    hand_mean = float(gray[y0:y1, x0:x1].mean())
    thr = max(18.0, hand_mean * 0.35)
    bw = (gray > thr).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels = cv2.connectedComponents(bw, connectivity=8)
    cxh = int(np.clip((x0 + x1) // 2, 0, W - 1))
    cyh = int(np.clip((y0 + y1) // 2, 0, H - 1))
    lab = labels[cyh, cxh]
    if lab == 0:  # box center not on the blob; take the box's dominant blob
        sub = labels[y0:y1, x0:x1]
        vals = sub[sub > 0]
        if len(vals) == 0:
            return None
        lab = int(np.bincount(vals).argmax())
    mask = labels == lab
    ex = int((x1 - x0) * 0.15)  # exclusion margin around the hand box
    ax0, ay0 = max(0, x0 - ex), max(0, y0 - ex)
    ax1, ay1 = min(W, x1 + ex), min(H, y1 + ex)
    arm = mask.copy()
    arm[ay0:ay1, ax0:ax1] = False
    # the hand blob may also connect to bright junk (table edges, cables);
    # keep only the piece that actually touches the hand box — the forearm —
    # not everything bright that happens to be connected.
    na, alabels = cv2.connectedComponents(arm.astype(np.uint8), connectivity=8)
    if na <= 1:
        return None
    ring = np.zeros_like(arm)
    g = 3  # pixels just outside the exclusion box
    ring[max(0, ay0 - g):ay1 + g, max(0, ax0 - g):ax1 + g] = True
    ring[ay0:ay1, ax0:ax1] = False
    touching = np.unique(alabels[ring & (alabels > 0)])
    if len(touching) == 0:
        return None
    sizes = [(np.sum(alabels == t), t) for t in touching]
    arm = alabels == max(sizes)[1]  # biggest piece touching the hand box
    ys, xs = np.nonzero(arm)
    if len(xs) < 300:  # no meaningful forearm visible
        return None
    # Root the forearm on the net's wrist (the shared joint). Direction is the
    # way the blob extends AWAY from the wrist into the arm; the wrist itself
    # is the net's, so hand and forearm share it by construction — they cannot
    # drift apart the way an independently-guessed wrist could.
    wrist = np.asarray(pts[0], np.float32)
    half = 0.5 * max(x1 - x0, y1 - y0)  # hand-box scale, sets the bands
    r = np.hypot(xs - wrist[0], ys - wrist[1])
    if r.min() > 1.4 * half + 10:  # blob never reaches the wrist: not the arm
        return None
    far = (r > 1.2 * half) & (r < 4.0 * half)  # forearm beyond the palm
    if far.sum() < 120:
        return None
    c_far = np.array([xs[far].mean(), ys[far].mean()], np.float32)
    elbow = c_far - wrist  # wrist -> elbow (proximal) direction
    n = float(np.linalg.norm(elbow))
    if n < 20:  # blob hugs the wrist with no reach: no direction
        return None
    elbow /= n
    d = -elbow  # arm -> hand convention (points toward the hand)
    # a real forearm OPPOSES the fingers: palm-forward is wrist -> middle
    # knuckle (joint 9). A bright blob in front of the hand is not an arm.
    fwd = np.asarray(pts[9], np.float32) - wrist
    fn = float(np.linalg.norm(fwd))
    if fn > 1e-3 and float((fwd / fn) @ elbow) > 0.25:
        return None
    conf = min(1.0, len(xs) / 3000.0) * min(1.0, n / (2.0 * half + 1e-3))
    if conf < 0.05:
        return None
    return {"dir": (float(d[0]), float(d[1])),
            "wrist2d": (float(wrist[0]), float(wrist[1])),
            "conf": float(conf)}


def _smooth(prev_filt, pts, confs, fps=30.0, min_cutoff=1.5, beta=0.01,
            d_cutoff=1.0):
    """One-Euro filter over the 21 joints (px). Adaptive low-pass: heavy
    smoothing when a joint is still, fast follow when it moves — the standard
    anti-jitter filter for hand tracking. Low-confidence observations are
    additionally down-weighted so a garbage heatmap can't teleport a joint.
    Returns (smoothed [21,2], new filter state)."""
    x = np.asarray(pts, np.float32)
    if prev_filt is None:
        return x, {"x": x, "dx": np.zeros_like(x)}
    dt = 1.0 / fps

    def alpha(cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    dx = (x - prev_filt["x"]) / dt
    dx_hat = alpha(d_cutoff) * dx + (1.0 - alpha(d_cutoff)) * prev_filt["dx"]
    a = alpha(min_cutoff + beta * np.abs(dx_hat))
    w = np.clip(np.asarray(confs, np.float32)[:, None] / 0.3, 0.15, 1.0)
    a = a * w
    x_hat = a * x + (1.0 - a) * prev_filt["x"]
    return x_hat, {"x": x_hat, "dx": dx_hat}


def run(det, kp, gray, args, state=None):
    """gray: uint8 HxW single eye image. Returns (annotated_bgr, info dict).
    state: mutable dict carried across frames (one per eye/view) enabling the
    keypoint net's temporal prior; pass None for stateless single images."""
    # ---- detection -------------------------------------------------------
    canvas, scale, ox, oy = letterbox(gray, DET_SIZE)
    inp = normalize_gray(canvas)[None, None]
    he, cx, cy, size = det.run(["hand_exists", "cx", "cy", "size"], {"inputImg": inp})

    prev_hands = state.get("hands", []) if state is not None else []

    # The detector has 2 hand slots (Mercury tracks left+right). Track the
    # best slot always; track the other too if it's confident AND not a
    # duplicate detection of the same hand.
    hands = []
    for slot in np.argsort(-he[0]):
        exists = float(he[0, slot])
        if exists < (SECOND_HAND_MIN if hands else HAND_MIN):
            break
        px = (cx[0, slot] * 0.5 + 0.5) * DET_SIZE  # [-1,1] -> canvas px
        py = (cy[0, slot] * 0.5 + 0.5) * DET_SIZE
        sz = abs(float(size[0, slot])) * DET_SIZE
        ex = (px - ox) / scale  # canvas -> original eye coords
        ey = (py - oy) / scale
        if hands:  # duplicate suppression: both slots locked on one hand
            pex, pey = hands[0]["center"]
            min_sep = 0.5 * max(hands[0]["size"], sz) / scale
            if np.hypot(ex - pex, ey - pey) < max(16.0, min_sep):
                continue
        # temporal prior: nearest last-frame hand, if it's plausibly the same
        # one (within a crop's width of the new detection)
        prev = None
        if prev_hands:
            dists = [np.hypot(ex - h["center"][0], ey - h["center"][1])
                     for h in prev_hands]
            i = int(np.argmin(dists))
            if dists[i] < max(32.0, (sz / scale) * args.crop_scale):
                prev = prev_hands.pop(i)

        # handedness: the keypoint net knows one hand; the other must be
        # mirrored. Auto-calibrate per track by trying both and keeping the
        # higher-confidence orientation; re-check periodically / when weak.
        if prev is not None and prev.get("flip_ttl", 0) > 0:
            fl = prev["flip"]
            pts, confs, box = _track_hand(kp, gray, ex, ey, sz, scale, args,
                                          prev_pts=prev["pts"], flip=fl)
            flip_ttl = prev["flip_ttl"] - 1
            if np.mean(confs) < 0.25:
                flip_ttl = 0  # weak: force a both-ways recheck next frame
        else:
            ppts = prev["pts"] if prev else None
            a = _track_hand(kp, gray, ex, ey, sz, scale, args,
                            prev_pts=ppts, flip=False)
            b = _track_hand(kp, gray, ex, ey, sz, scale, args,
                            prev_pts=ppts, flip=True)
            fl = bool(np.mean(b[1]) > np.mean(a[1]))
            pts, confs, box = b if fl else a
            flip_ttl = 60  # trust the choice for ~60 frames
        # anti-jitter: One-Euro smooth the joints across frames
        sm, filt = _smooth(prev.get("filt") if prev else None, pts, confs)
        pts = [(float(p[0]), float(p[1])) for p in sm]
        hands.append({"exists": exists, "pts": pts, "confs": confs,
                      "box": box, "center": (ex, ey), "size": sz,
                      "flip": fl, "flip_ttl": flip_ttl, "filt": filt,
                      "half": (box[2] - box[0]) * 0.5})

    # --- keypoint coasting: keep tracks the detector lost this frame --------
    # The detector sees the whole frame letterboxed to 160x160 (a 640x240
    # frame becomes a 160x60 strip), so a hand past ~150 mm is only ~12 px to
    # it and detection dies long before the keypoint net would: a native-res
    # crop of the same hand is still ~40+ px. Mercury detects only to
    # (re)acquire and otherwise crops around the previous frame's keypoints;
    # do the same — crop at the old joints' centroid, keep the track while
    # keypoint confidence holds up.
    for ph in prev_hands:  # entries the detection loop didn't consume
        if len(hands) >= 2:
            break
        p = np.asarray(ph["pts"], np.float32)
        ex, ey = float(p[:, 0].mean()), float(p[:, 1].mean())
        half = ph.get("half", 0.0)
        if half <= 8.0:
            continue
        if any(np.hypot(ex - h["center"][0], ey - h["center"][1])
               < max(40.0, half) for h in hands):
            continue  # the detector re-found this hand; nothing to coast
        sz = half * scale / (0.5 * args.crop_scale)  # invert _track_hand calc
        if ph.get("flip_ttl", 0) > 0:
            fl = ph["flip"]
            pts, confs, box = _track_hand(kp, gray, ex, ey, sz, scale, args,
                                          prev_pts=ph["pts"], flip=fl)
            flip_ttl = ph["flip_ttl"] - 1
        else:
            a = _track_hand(kp, gray, ex, ey, sz, scale, args,
                            prev_pts=ph["pts"], flip=False)
            b = _track_hand(kp, gray, ex, ey, sz, scale, args,
                            prev_pts=ph["pts"], flip=True)
            fl = bool(np.mean(b[1]) > np.mean(a[1]))
            pts, confs, box = b if fl else a
            flip_ttl = 60
        conf = float(np.mean(confs))
        if conf < COAST_MIN:
            continue  # keypoints gone too: the hand really left
        sm, filt = _smooth(ph.get("filt"), pts, confs)
        pts = [(float(q[0]), float(q[1])) for q in sm]
        # adapt the crop to the hand's current apparent size (it shrinks as
        # the hand rises) instead of freezing at the last detection's size —
        # keeps the 128x128 crop budget spent on hand, not background.
        # Blended with the previous size so finger curls don't pump the box.
        q = np.asarray(pts, np.float32)
        ext = float(max(np.ptp(q[:, 0]), np.ptp(q[:, 1])))
        half_kp = max(20.0, 0.5 * ext * args.crop_scale * 1.15)
        half_next = 0.7 * half + 0.3 * half_kp
        hands.append({"exists": conf, "pts": pts, "confs": confs,
                      "box": box, "center": (ex, ey), "size": sz * scale,
                      "flip": fl, "flip_ttl": flip_ttl, "filt": filt,
                      "half": half_next, "coast": True})

    if state is not None:  # remember this frame's joints for the next one
        state["hands"] = [{"center": h["center"], "pts": h["pts"],
                           "flip": h["flip"], "flip_ttl": h["flip_ttl"],
                           "filt": h["filt"], "half": h["half"]}
                          for h in hands]

    # ---- forearm: anatomical anchor for the kinematic fit -----------------
    for h in hands:
        h["arm"] = detect_arm(gray, h["box"], h["pts"])

    # ---- draw ------------------------------------------------------------
    vis = cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=args.gain), cv2.COLOR_GRAY2BGR)
    for h in hands:
        x0, y0, x1, y1 = h["box"]
        # green box = detector-confirmed (solid lock), amber = coasting on
        # keypoints only (detector can't see it — far hand / weakening)
        bcol = (0, 180, 220) if h.get("coast") else (0, 200, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), bcol, 1)
        if h["arm"]:  # magenta: detected forearm axis + wrist estimate
            wx, wy = h["arm"]["wrist2d"]
            dx, dy = h["arm"]["dir"]
            cv2.line(vis, (int(wx - dx * 120), int(wy - dy * 120)),
                     (int(wx), int(wy)), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(vis, (int(wx), int(wy)), (255, 0, 255),
                           cv2.MARKER_CROSS, 8, 1)
        pts = h["pts"]
        for a, b in BONES:
            col = FINGER_COLORS[finger_of(b)]
            cv2.line(vis, (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])), col, 1, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 2, FINGER_COLORS[finger_of(j)], -1,
                       cv2.LINE_AA)
    top_score = float(he[0].max())
    if not hands:
        cv2.putText(vis, f"no hand (best={top_score:.2f} < {HAND_MIN})",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        return vis, {"exists": top_score, "kp_conf": 0.0, "hands": 0,
                     "joints": []}
    best = hands[0]
    label = " + ".join(f"{h['exists']:.2f}" for h in hands)
    cv2.putText(vis, f"hands={len(hands)} exists={label} "
                f"kp_conf={np.mean(best['confs']):.2f}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return vis, {"exists": best["exists"], "kp_conf": float(np.mean(best["confs"])),
                 "center": best["center"], "size": best["size"],
                 "hands": len(hands), "expo": _expo_health(gray, best["box"]),
                 # full per-hand results, machine-readable (for recordings)
                 "joints": [{"exists": h["exists"],
                             "pts": [[round(float(x), 1), round(float(y), 1)]
                                     for x, y in h["pts"]],
                             "confs": [round(float(c), 3) for c in h["confs"]],
                             "arm": ({"dir": [round(v, 3) for v in h["arm"]["dir"]],
                                      "wrist2d": [round(v, 1) for v in h["arm"]["wrist2d"]],
                                      "conf": round(h["arm"]["conf"], 2)}
                                     if h["arm"] else None)}
                            for h in hands]}


def _expo_health(gray, box):
    """Exposure health of the hand region the net actually saw: brightest-part
    p95 and the % of hand pixels clipped to white. Measured on the raw image
    (NOT the display gain), so it reflects the signal the net is fed. p95 ~200
    with clip ~0 is the sweet spot; clip>0 means shading is being lost to white,
    p95<120 means the hand is too dim. Returns None if the box is unusable."""
    H, W = gray.shape
    x0, y0, x1, y1 = (max(0, int(box[0])), max(0, int(box[1])),
                      min(W, int(box[2])), min(H, int(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = gray[y0:y1, x0:x1]
    thr = max(20.0, float(roi.mean()))  # hand = the brighter-than-mean pixels
    hand = roi[roi > thr]
    if hand.size < 50:
        return None
    return {"p95": float(np.percentile(hand, 95)),
            "clip": float(100.0 * np.mean(hand >= 250)),
            "med": float(np.median(hand))}


def load_gray(path, args):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {path}")
    if args.rot:
        g = np.rot90(g, k=args.rot // 90)
    if args.flip:
        g = cv2.flip(g, 1)
    return np.ascontiguousarray(g)


def load_models(models_dir=MODELS):
    # Cap onnxruntime's thread pool: by default it grabs every core, which
    # starves the USB drain thread in openleap and tears frames (seen as
    # short/128 spikes whenever the skeleton is enabled). These nets are tiny;
    # two threads is plenty.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    det = ort.InferenceSession(os.path.join(models_dir, "grayscale_detection_160x160.onnx"),
                               sess_options=so, providers=["CPUExecutionProvider"])
    kp = ort.InferenceSession(os.path.join(models_dir, "grayscale_keypoint_jan18.onnx"),
                              sess_options=so, providers=["CPUExecutionProvider"])
    return det, kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="single image (pgm/png), one eye")
    ap.add_argument("--dir", help="process all *_left.pgm in a directory")
    ap.add_argument("--rot", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--gain", type=float, default=4.0, help="display brightness gain")
    ap.add_argument("--crop-scale", type=float, default=1.3, dest="crop_scale")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    det = ort.InferenceSession(os.path.join(MODELS, "grayscale_detection_160x160.onnx"),
                               providers=["CPUExecutionProvider"])
    kp = ort.InferenceSession(os.path.join(MODELS, "grayscale_keypoint_jan18.onnx"),
                              providers=["CPUExecutionProvider"])

    paths = []
    if args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, "*_left.pgm")))
    elif args.image:
        paths = [args.image]
    else:
        ap.error("give an image or --dir")

    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    for p in paths:
        gray = load_gray(p, args)
        vis, info = run(det, kp, gray, args)
        out = args.out or os.path.join(outdir, os.path.splitext(os.path.basename(p))[0] + "_skel.png")
        cv2.imwrite(out, vis)
        where = (f"center=({info['center'][0]:.0f},{info['center'][1]:.0f})"
                 if info.get("hands") else "no hand")
        print(f"{os.path.basename(p):40s} exists={info['exists']:.2f} "
              f"kp_conf={info['kp_conf']:.2f} {where} -> {out}")


if __name__ == "__main__":
    main()
