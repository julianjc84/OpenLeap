#!/usr/bin/env python3
"""Kinematic hand-model fit: constrain the 21 triangulated 3D joints to an
anatomically possible hand pose (kine_lm's architecture, scipy's solver).

Instead of trusting 21 independent points, parameterize the hand as a pose:

  wrist position (3) + wrist orientation (3, rotation vector)
  + per finger [thumb,index,middle,ring,little]:
      abduct (sideways splay) + 3 flex angles (knuckle, mid, tip)   = 5*4
  -> 26 degrees of freedom

with the user's measured bone lengths held fixed (bone_lengths.npz, median
over confident frames) and joint angles bounded to anatomical ranges. Each
frame, scipy.optimize.least_squares finds the pose whose forward kinematics
best explain the observed joints (confidence-weighted), warm-started from the
previous frame.

What this buys over raw triangulation: bones cannot stretch (raw data wobbles
~14%), the thumb cannot land on the pinky side, occluded joints are inferred
from hand geometry instead of being garbage, and the pose's joint ANGLES come
out for free (gesture detection wants angles, not pixels).

Joint order matches skeleton.py / Mercury joints_ml_to_xr:
  0 wrist; 1-4 thumb(mcp,prox,dist,tip); then index/middle/ring/little
  (prox,int,dist,tip) -> bones (0,1)(1,2)(2,3)(3,4),(0,5)(5,6)(6,7)(7,8),...
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# finger chains: (first-joint index, [meta, b1, b2, b3] bone-length indices)
FINGERS = [(1, 0), (5, 4), (9, 8), (13, 12), (17, 16)]  # (joint0, bone0)
# rest direction of each finger's metacarpal in the palm frame (y = fingers
# forward, x = thumb side for a RIGHT hand, z = palm normal). Splay angles
# eyeballed from hand anatomy; the abduct DOF absorbs per-user variation.
REST_SPLAY_DEG = {0: 55.0, 1: 15.0, 2: 0.0, 3: -12.0, 4: -25.0}  # thumb..little
THUMB_DROP = -0.45  # thumb metacarpal points out of the palm plane

# bounds (radians): [abduct, flex1, flex2, flex3]
FINGER_LO = np.array([-0.35, -0.35, -0.10, -0.10])
FINGER_HI = np.array([0.35, 1.75, 1.95, 1.60])
THUMB_LO = np.array([-0.90, -0.60, -0.30, -0.30])
THUMB_HI = np.array([0.90, 1.20, 1.10, 1.40])


def _finger_frame(fi, handed):
    """Rest direction d and flex axis x for finger fi in the palm frame."""
    a = np.deg2rad(REST_SPLAY_DEG[fi])
    d = np.array([np.sin(a) * handed, np.cos(a), 0.0])
    if fi == 0:  # thumb: out of plane
        d = np.array([np.sin(a) * handed, np.cos(a), THUMB_DROP])
        d /= np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, d)
    x /= np.linalg.norm(x)
    return d, x


def _rot(axis, angle):
    return Rotation.from_rotvec(axis * angle).as_matrix()


def forward(params, lengths, handed):
    """params [26] -> joints [21,3] in camera mm. lengths: [20] bone mm."""
    t = params[0:3]
    Rw = Rotation.from_rotvec(params[3:6]).as_matrix()
    joints = np.zeros((21, 3))
    joints[0] = t
    z = np.array([0.0, 0.0, 1.0])
    for fi, (j0, b0) in enumerate(FINGERS):
        ab, f1, f2, f3 = params[6 + fi * 4: 10 + fi * 4]
        d, x = _finger_frame(fi, handed)
        meta, l1, l2, l3 = lengths[b0:b0 + 4]
        p = d * meta  # metacarpal: rigid in the palm frame
        joints[j0] = t + Rw @ p
        Rc = _rot(z, ab * handed) @ _rot(x, f1)
        p = p + (Rc @ d) * l1
        joints[j0 + 1] = t + Rw @ p
        Rc = Rc @ _rot(x, f2)
        p = p + (Rc @ d) * l2
        joints[j0 + 2] = t + Rw @ p
        Rc = Rc @ _rot(x, f3)
        p = p + (Rc @ d) * l3
        joints[j0 + 3] = t + Rw @ p
    return joints


def neutral_params(obs):
    """Initial guess: wrist at observed wrist, palm facing the camera,
    fingers pointing the way the observed middle finger points."""
    p = np.zeros(26)
    p[0:3] = obs[0]
    fwd = obs[10] - obs[0]  # wrist -> middle intermediate
    n = np.linalg.norm(fwd)
    if n > 1e-6:
        fwd = fwd / n
        # rotation taking +y to fwd (palm normal free; LM refines it)
        rot, _ = Rotation.align_vectors([fwd], [[0.0, 1.0, 0.0]])
        p[3:6] = rot.as_rotvec()
    p[6::4] = 0.0   # abducts
    p[7::4] = 0.3   # slight rest curl
    return p


def _bounds():
    lo = np.full(26, -np.inf)
    hi = np.full(26, np.inf)
    lo[3:6], hi[3:6] = -2 * np.pi, 2 * np.pi
    for fi in range(5):
        s = 6 + fi * 4
        lo[s:s + 4] = THUMB_LO if fi == 0 else FINGER_LO
        hi[s:s + 4] = THUMB_HI if fi == 0 else FINGER_HI
    return lo, hi


class HandFitter:
    """Per-hand stateful fitter: warm-starts from the previous frame and
    auto-detects handedness on the first confident fit."""

    def __init__(self, lengths, temporal_weight=2.0):
        self.lengths = np.asarray(lengths, float)
        self.prev = None
        self.handed = None  # +1 right, -1 left (auto-detected)
        self.temporal_weight = temporal_weight
        self._lo, self._hi = _bounds()

    def _solve(self, obs, w, x0, handed, arm=None, cam=None):
        prev = self.prev
        # forearm anchor (2D, left eye): the wrist is now shared with the hand
        # (the net's joint 0), so the forearm's job here is purely the wrist
        # ORIENTATION — the anatomical truth that the palm continues the
        # forearm direction. The wrist POSITION is already an observation.
        arm_dir = None
        arm_w = 0.0
        if arm is not None and cam is not None:
            arm_dir = np.asarray(arm["dir"], float)
            arm_w = float(arm["conf"])
        fx, fy, cx, cy = cam if cam is not None else (1, 1, 0, 0)

        def project(P):
            return np.array([P[0] * fx / P[2] + cx, P[1] * fy / P[2] + cy])

        def resid(p):
            J = forward(p, self.lengths, handed)
            r = ((J - obs) * w[:, None]).ravel()
            parts = [r]
            if prev is not None:
                # gentle temporal prior on the angle DOFs only. (A stiffer
                # prior covering position/orientation was tried and made
                # everything WORSE — it slows convergence within the
                # iteration budget and the solution oscillates. Output
                # smoothness is the One-Euro filter's job, not the prior's.)
                parts.append(self.temporal_weight * (p[6:] - prev[6:]) * 0.1)
            if arm_w > 0.0 and J[0][2] > 1.0:
                wp = project(J[0])           # fitted wrist in left-eye px
                kp = project(J[9])           # middle knuckle
                v = kp - wp
                n = np.linalg.norm(v)
                if n > 1e-3:
                    v = v / n
                    # palm-forward vs forearm direction: penalize the sine of
                    # the angle (zero when aligned), and heavily penalize
                    # pointing backwards into the arm
                    cross = v[0] * arm_dir[1] - v[1] * arm_dir[0]
                    dot = float(v @ arm_dir)
                    parts.append(np.array([
                        arm_w * 15.0 * cross,
                        arm_w * 25.0 * max(0.0, -dot),
                    ]))
            return np.concatenate(parts)

        return least_squares(resid, x0, bounds=(self._lo, self._hi),
                             method="trf", max_nfev=60, xtol=1e-3, ftol=1e-4)

    def fit(self, xyz, valid, confs=None, arm=None, cam=None):
        """xyz [21,3] mm, valid [21] bool, confs [21] (optional).
        arm: detect_arm() result for this hand (left eye); cam: (fx,fy,cx,cy)
        of the rectified left eye — both optional, enable the forearm anchor.
        Returns dict: joints [21,3] (constrained), params, angles, rms_mm."""
        obs = np.asarray(xyz, float)
        w = np.asarray(valid, float)
        if confs is not None:
            w = w * np.clip(np.asarray(confs, float) / 0.3, 0.2, 1.0)
        if w.sum() < 4:  # not enough evidence to fit anything
            return None

        x0 = self.prev if self.prev is not None else neutral_params(obs)
        if self.handed is None:
            # first confident frame: try both handednesses, keep the better
            a = self._solve(obs, w, neutral_params(obs), +1, arm, cam)
            b = self._solve(obs, w, neutral_params(obs), -1, arm, cam)
            self.handed = 1 if a.cost <= b.cost else -1
            res = a if a.cost <= b.cost else b
        else:
            res = self._solve(obs, w, x0, self.handed, arm, cam)
        self.prev = res.x

        joints = forward(res.x, self.lengths, self.handed)
        err = np.linalg.norm(joints - obs, axis=1)
        m = w > 0
        rms = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")
        angles = res.x[6:].reshape(5, 4)  # [finger][abduct,f1,f2,f3]
        return {"joints": joints, "params": res.x.copy(), "angles": angles,
                "rms_mm": rms, "handed": self.handed}

    def reset(self):
        self.prev = None
        self.handed = None


def load_lengths(path):
    d = np.load(path, allow_pickle=True)
    return d["lengths"]


# bone connectivity (joint pairs) + names — MUST match skeleton.BONES order so
# the saved lengths line up with forward()'s bone indexing.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
BONE_NAMES = ["thumb_meta", "thumb_1", "thumb_2", "thumb_3",
              "index_meta", "index_1", "index_2", "index_3",
              "middle_meta", "middle_1", "middle_2", "middle_3",
              "ring_meta", "ring_1", "ring_2", "ring_3",
              "little_meta", "little_1", "little_2", "little_3"]


class BoneCalibrator:
    """Measure a user's bone lengths from RAW triangulated 3D joints.

    A bone's length is pose-invariant, so we sample ``||jointA - jointB||`` over
    many frames and take the per-bone median — but only from frames where BOTH
    endpoints are valid and confident (occluded/curled joints triangulate
    badly, biasing the estimate). Each bone fills independently; the slowest
    bone (usually a thumb joint or an inner knuckle) gates ``done()``, which
    naturally nudges the user to show the whole hand from several angles.

    Feed it the RAW triangulation, never the kinematic fit's output: the fit
    already assumes bone lengths, so measuring from it would be circular.
    """

    def __init__(self, target=300, conf_min=0.35):
        self.target = int(target)
        self.conf_min = float(conf_min)
        self.samples = [[] for _ in BONES]
        self.frames = 0

    def add(self, xyz, valid, confs):
        xyz = np.asarray(xyz, float)
        valid = np.asarray(valid, bool)
        confs = np.asarray(confs, float)
        self.frames += 1
        for bi, (a, b) in enumerate(BONES):
            if (valid[a] and valid[b]
                    and confs[a] >= self.conf_min and confs[b] >= self.conf_min):
                self.samples[bi].append(float(np.linalg.norm(xyz[a] - xyz[b])))

    def min_count(self):
        return min(len(s) for s in self.samples)

    def progress(self):
        return min(1.0, self.min_count() / max(1, self.target))

    def done(self):
        return self.min_count() >= self.target

    def result(self):
        """Per-bone (median_mm, mad_mm, count); NaN length where unsampled."""
        out = []
        for s in self.samples:
            if s:
                a = np.asarray(s)
                med = float(np.median(a))
                out.append((med, float(np.median(np.abs(a - med))), len(a)))
            else:
                out.append((float("nan"), float("nan"), 0))
        return out

    def save(self, path):
        lengths = np.array([r[0] for r in self.result()], float)
        np.savez(path, bones=np.array(BONES), lengths=lengths,
                 names=np.array(BONE_NAMES))
        return lengths
