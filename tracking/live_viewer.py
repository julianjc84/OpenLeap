#!/usr/bin/env python3
"""Live skeleton viewer for the open Leap Motion stack.

Launches `openleap stream` (which brings the device up and writes raw frames
to stdout), reads frames in a thread, runs Mercury's ONNX skeleton inference in
a worker thread, and shows a live GTK4 window: the IR image with the 21-joint
skeleton drawn on top, plus on-screen controls (rotation, brightness gain,
crop scale) and live readouts (FPS, detection confidence). Move your hand and
watch what the tracker does — then tell me what you see.

Capture needs root (USB), the GUI needs your session, so openleap is launched
via `pkexec` and its stdout is piped here. One polkit prompt at startup.

Run:  ./run_live.sh        (wrapper that points at the venv + binary)
  or: .venv/bin/python tracking/live_viewer.py
"""

import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton import run as run_skeleton, load_models, triangulate  # noqa: E402
from skeleton import BONES as SKEL_BONES  # noqa: E402
import calibration as calib_mod  # noqa: E402

try:
    from kinematic import HandFitter, load_lengths as load_bone_lengths
    from kinematic import BoneCalibrator
except Exception as _e:  # missing scipy etc. — viewer still works without
    HandFitter = None
    BoneCalibrator = None
    print(f"kinematic module unavailable ({_e})", file=sys.stderr)

import gi  # noqa: E402

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import GLib, Gdk, Gtk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEAP_OPEN = os.path.join(HERE, "..", "rust-driver", "target", "debug", "openleap")
SEQ = os.path.join(HERE, "..", "rust-driver", "bringup_seq.txt")
FRAME_BYTES = 1280 * 240
EYE_W, EYE_H, INTER_W = 640, 240, 1280


class FrameReader(threading.Thread):
    """Reads [b'LFRM'][u32 counter][FRAME_BYTES] records from a pipe."""

    def __init__(self, proc):
        super().__init__(daemon=True)
        self.proc = proc
        self.lock = threading.Lock()
        self.latest = None  # (counter, np.uint8 [240,1280])
        self.running = True
        self.bytes_seen = 0
        # frame-integrity stats (for glitch diagnosis)
        self.total = 0       # frames successfully parsed
        self.drops = 0       # counter gaps (frames openleap skipped)
        self.resyncs = 0     # times we had to hunt for the magic again
        self._prev_counter = None

    def _read_exact(self, n):
        out = bytearray()
        while len(out) < n and self.running:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                return None
            out += chunk
        return bytes(out)

    def run(self):
        f = self.proc.stdout
        while self.running:
            # resync on magic
            b = f.read(1)
            if not b:
                break
            if b != b"L":
                with self.lock:
                    self.resyncs += 1
                continue
            rest = self._read_exact(3)
            if rest != b"FRM":
                with self.lock:
                    self.resyncs += 1
                continue
            cnt = self._read_exact(4)
            if cnt is None:
                break
            data = self._read_exact(FRAME_BYTES)
            if data is None:
                break
            counter = struct.unpack("<I", cnt)[0]
            arr = np.frombuffer(data, np.uint8).reshape(EYE_H, INTER_W)
            with self.lock:
                self.latest = (counter, arr)
                self.bytes_seen += FRAME_BYTES
                self.total += 1
                if self._prev_counter is not None:
                    gap = counter - self._prev_counter - 1
                    if gap > 0:
                        self.drops += gap
                self._prev_counter = counter

    def stats(self):
        with self.lock:
            return self.total, self.drops, self.resyncs

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False


class Inference(threading.Thread):
    """Pulls the latest frame, runs the skeleton, stores an annotated RGB image."""

    def __init__(self, reader, controls):
        super().__init__(daemon=True)
        self.reader = reader
        self.controls = controls
        self.det, self.kp = load_models()
        calib_path = os.path.join(HERE, "calib.npz")
        try:
            self.cal = calib_mod.load_npz(calib_path)
            print(f"loaded undistortion maps from {calib_path}", file=sys.stderr)
        except Exception as e:
            self.cal = None
            print(f"no calibration ({e}); undistort disabled", file=sys.stderr)
            controls.undistort = False
        self.lock = threading.Lock()
        self.annotated = None  # RGB uint8 [H,W,3]
        self.info = {}
        self.running = True
        self.n = 0
        self._last_counter = -1
        self.states = {}  # per-eye temporal state for the keypoint prior
        self.fitters = {}  # per-hand kinematic fitters (keyed by pair index)
        self.bone_lengths = None
        self.bone_sampler = None  # set to a BoneCalibrator to sample live
        if HandFitter is not None:
            try:
                self.bone_lengths = load_bone_lengths(
                    os.path.join(HERE, "bone_lengths.npz"))
            except Exception as e:
                print(f"no bone lengths ({e}); kinematic fit disabled",
                      file=sys.stderr)
                controls.kinematic = False

    def run(self):
        while self.running:
            item = self.reader.get()
            if item is None or item[0] == self._last_counter:
                threading.Event().wait(0.005)
                continue
            self._last_counter, frame = item
            args = SimpleNamespace(gain=self.controls.gain,
                                   crop_scale=self.controls.crop_scale)

            def proc(eye):
                img = orient(frame, self.controls, self.cal, eye=eye)
                if not self.controls.skeleton:
                    # raw frame-inspection mode: just brighten + grayscale->RGB,
                    # no ONNX (frees the CPU and shows the unaltered IR pixels).
                    g = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=self.controls.gain),
                                     cv2.COLOR_GRAY2RGB)
                    return g, {"exists": 0.0, "kp_conf": 0.0}
                if self.controls.temporal:
                    st = self.states.setdefault(eye, {})
                else:
                    st = None
                    self.states.clear()
                try:
                    vis, info = run_skeleton(self.det, self.kp, img, args, state=st)
                    if self.controls.record:
                        self._record(eye, img, info)
                    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), info
                except Exception as e:  # keep the viewer alive on a bad frame
                    g = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=self.controls.gain),
                                     cv2.COLOR_GRAY2RGB)
                    return g, {"exists": 0.0, "kp_conf": 0.0, "err": str(e)}

            if self.controls.eye == 2:  # both eyes side by side
                (l_rgb, l_info), (r_rgb, r_info) = proc(0), proc(1)
                div = np.full((l_rgb.shape[0], 4, 3), 60, np.uint8)
                rgb = np.hstack([l_rgb, div, r_rgb])
                info = {"exists": max(l_info["exists"], r_info["exists"]),
                        "kp_conf": max(l_info["kp_conf"], r_info["kp_conf"])}
                tri = self._triangulate(l_info, r_info)
                if tri:
                    info.update(tri)
                    rgb = self._draw_fit(np.ascontiguousarray(rgb), tri)
                    if self.controls.record:
                        self._record_tri(tri)
            else:
                rgb, info = proc(self.controls.eye)
            with self.lock:
                self.annotated = np.ascontiguousarray(rgb)
                self.info = info
                self.n += 1

    def _triangulate(self, l_info, r_info):
        """Stereo-triangulate per-joint 3D (mm) from the two eyes' skeletons.
        Needs a rectified calibration and the native geometry (undistort on,
        no rotation/flips — the rectified maps ARE the undistort step)."""
        c = self.controls
        if (self.cal is None or not self.cal.get("rectified")
                or not c.undistort or c.rot or c.flip_h or c.flip_v):
            return None
        hl, hr = l_info.get("joints", []), r_info.get("joints", [])
        if not hl or not hr:
            return None
        # pair hands across eyes by x-order (disparity shifts x only slightly)
        key = lambda h: float(np.mean([p[0] for p in h["pts"]]))  # noqa: E731
        hl = sorted(hl, key=key)
        hr = sorted(hr, key=key)
        hands3d = []
        best_raw = None  # most-complete hand this frame, for bone calibration
        for hi, (L, R) in enumerate(zip(hl, hr)):
            xyz, valid = triangulate(L["pts"], R["pts"], L["confs"], R["confs"],
                                     self.cal)
            if not valid.any():
                continue
            confs = np.minimum(np.asarray(L["confs"], float),
                               np.asarray(R["confs"], float))
            vc = int(valid.sum())
            if best_raw is None or vc > best_raw[0]:
                best_raw = (vc, xyz, valid, confs)  # RAW, pre-fit (not circular)
            entry = {}
            fitted = None
            if (self.controls.kinematic and self.bone_lengths is not None
                    and HandFitter is not None):
                fitter = self.fitters.setdefault(hi,
                                                 HandFitter(self.bone_lengths))
                cam = (self.cal["fx"], self.cal["fy"],
                       self.cal["cx"], self.cal["cy"])
                r = fitter.fit(xyz, valid, confs, arm=L.get("arm"), cam=cam)
                if r:
                    fitted = r["joints"]
                    entry["rms_mm"] = round(r["rms_mm"], 1)
            use = fitted if fitted is not None else xyz
            pinch = float(np.linalg.norm(use[4] - use[8]))  # thumbtip-indextip
            depth = float(np.nanmedian(use[:, 2]))
            entry.update({"pinch_mm": round(pinch, 1),
                          "depth_mm": round(depth, 1),
                          "valid": int(valid.sum()),
                          "kin": fitted is not None,
                          "xyz": [[round(float(v), 1) for v in p] for p in use]})
            hands3d.append(entry)
        sampler = self.bone_sampler
        if sampler is not None and best_raw is not None:
            sampler.add(best_raw[1], best_raw[2], best_raw[3])
        return {"hands3d": hands3d} if hands3d else None

    def reload_bone_lengths(self, lengths):
        """Swap in freshly-measured bone lengths and drop the warm-started
        fitters so they re-converge on the new model."""
        self.bone_lengths = np.asarray(lengths, float)
        self.fitters.clear()

    def _draw_fit(self, rgb, tri):
        """Overlay the kinematic skeleton (cyan), re-projected into the LEFT
        eye, on the combined display image — so raw (colored) vs constrained
        (cyan) is visible live."""
        if self.cal is None or not self.cal.get("rectified"):
            return rgb
        fx, fy = self.cal["fx"], self.cal["fy"]
        cx, cy = self.cal["cx"], self.cal["cy"]
        for h in tri["hands3d"]:
            if not h.get("kin"):
                continue
            pts = []
            for x, y, z in h["xyz"]:
                pts.append((int(x * fx / z + cx), int(y * fy / z + cy))
                           if z > 1.0 else None)
            for a, b in SKEL_BONES:
                if pts[a] and pts[b]:
                    cv2.line(rgb, pts[a], pts[b], (0, 255, 255), 1, cv2.LINE_AA)
        return rgb

    def _record_tri(self, tri):
        d = self.controls.record_dir
        if not d:
            return
        rec = {"frame": self._last_counter, "eye": "3d",
               "hands3d": tri["hands3d"]}
        with open(os.path.join(d, "results.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _record(self, eye, img, info):
        """Save the model's exact input frame + its full output, for offline
        analysis. PGM per eye per frame + one JSONL of results."""
        d = self.controls.record_dir
        if not d:
            return
        fn = f"{self._last_counter:06d}_{'LRB'[eye]}.pgm"
        cv2.imwrite(os.path.join(d, fn), img)
        c = self.controls
        rec = {"frame": self._last_counter, "eye": eye, "file": fn,
               "exists": round(info.get("exists", 0.0), 3),
               "kp_conf": round(info.get("kp_conf", 0.0), 3),
               "hands": info.get("hands", 0), "joints": info.get("joints", []),
               "settings": {"rot": c.rot, "flip_h": c.flip_h, "flip_v": c.flip_v,
                            "undistort": c.undistort, "crop": round(c.crop_scale, 2),
                            "temporal": c.temporal}}
        with open(os.path.join(d, "results.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def get(self):
        with self.lock:
            if self.annotated is None:
                return None, None, 0
            return self.annotated, dict(self.info), self.n

    def stop(self):
        self.running = False


class Controls:
    # defaults = the production 3D-tracking config (what we use all the time):
    # both eyes + undistort(rectified) + skeleton + temporal = live 3D readout
    rot = 0          # neutral: no rotation
    gain = 2.0
    crop_scale = 1.5
    eye = 2          # 0 = left, 1 = right, 2 = both (needed for 3D)
    flip_h = False   # mirror left-right
    flip_v = False   # mirror top-bottom
    undistort = True   # rectified maps; required for triangulation
    skeleton = True    # tracking on at startup
    kinematic = True   # fit the 26-DOF hand model to the triangulated joints
    temporal = True    # feed last frame's joints to the keypoint net (Mercury prior)
    record = False     # save frames + per-frame results for offline analysis
    record_dir = None


def orient(frame, c, cal=None, eye=None):
    """eye select -> undistort (native frame) -> flips -> rotation."""
    e = c.eye if eye is None else eye
    img = np.ascontiguousarray(frame[:, e::2])
    if c.undistort and cal is not None:
        cam = cal["left" if e == 0 else "right"]
        img = cv2.remap(img, cam["map1"], cam["map2"], cv2.INTER_LINEAR)
    if c.flip_h:
        img = cv2.flip(img, 1)
    if c.flip_v:
        img = cv2.flip(img, 0)
    if c.rot:
        img = np.rot90(img, k=c.rot // 90)
    return np.ascontiguousarray(img)


class Viewer(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.niri.LeapLiveSkeleton")
        self.controls = Controls()
        self.proc = None
        self.reader = None
        self.infer = None
        self._fps_t = GLib.get_monotonic_time()
        self._fps_n = 0
        self._fps = 0.0
        self._last_shown_n = -1
        # device (stream) fps: frames arriving over the pipe per second
        self._dev_fps = 0.0
        self._dev_prev = 0
        self._dev_t = GLib.get_monotonic_time()
        # device exposure state, parsed from openleap's stderr diag line. The
        # driver owns exposure (incl. whatever auto-exposure picks), so this is
        # the ground truth — what the SENSOR actually used, not what we asked.
        self._dev_exp = None    # microseconds actually applied
        self._dev_ae = None     # True if auto-exposure is driving it
        self._dev_mean = None   # device's whole-frame mean brightness
        self._bone_cal = None   # active BoneCalibrator while sampling
        self._cal_btn = None    # the "calibrate bones" button (label toggles)

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self, title="Leap Live Skeleton (open stack)")
        win.set_default_size(900, 560)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        win.set_child(box)

        self.pic = Gtk.Picture()
        self.pic.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.pic.set_vexpand(True)
        self.pic.set_hexpand(True)
        box.append(self.pic)

        self.status = Gtk.Label(label="starting stream… (authorize the prompt)")
        self.status.set_xalign(0.0)
        box.append(self.status)

        # ---- control panel: 2x2 grid of labeled groups ----------------------
        #   View     | Tracking      View   = how the image is presented
        #   Device   | Capture       Device = hardware (sent to openleap stdin)
        def group(title, grid, col, row):
            frame = Gtk.Frame(label=title)
            inner = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            inner.set_margin_start(8); inner.set_margin_end(8)
            inner.set_margin_top(4); inner.set_margin_bottom(6)
            frame.set_child(inner)
            frame.set_hexpand(True)
            grid.attach(frame, col, row, 1, 1)
            return inner

        grid = Gtk.Grid(column_spacing=8, row_spacing=6)
        grid.set_margin_start(8); grid.set_margin_end(8); grid.set_margin_bottom(8)
        box.append(grid)

        # ---- View: eye, rotation, flips, gain --------------------------------
        view = group("View", grid, 0, 0)
        view.append(Gtk.Label(label="eye"))
        eye = Gtk.DropDown.new_from_strings(["left", "right", "both"])
        eye.set_tooltip_text("Which camera to show. 'both' runs tracking on both eyes\n"
                             "and enables 3D triangulation (pinch/depth in mm).")
        eye.set_selected(self.controls.eye)
        eye.connect("notify::selected", lambda d, _:
                    setattr(self.controls, "eye", d.get_selected()))
        view.append(eye)

        view.append(Gtk.Label(label="rotation"))
        rot = Gtk.DropDown.new_from_strings(["0", "90", "180", "270"])
        rot.set_tooltip_text("Rotate the view. The neural nets see exactly this view,\n"
                             "so rotation changes tracking, not just display.")
        rot.set_selected(0)  # 0 default (neutral)
        rot.connect("notify::selected", lambda d, _:
                    setattr(self.controls, "rot", int([0, 90, 180, 270][d.get_selected()])))
        view.append(rot)

        # CheckButtons, not ToggleButtons: an explicit checkmark beats GTK4's
        # subtle pressed-in shading for telling on from off at a glance.
        fh = Gtk.CheckButton(label="flip H")
        fh.set_tooltip_text("Mirror left-right (display AND model input).")
        fh.connect("toggled", lambda b: setattr(self.controls, "flip_h", b.get_active()))
        view.append(fh)
        fv = Gtk.CheckButton(label="flip V")
        fv.set_tooltip_text("Mirror top-bottom (display AND model input).")
        fv.connect("toggled", lambda b: setattr(self.controls, "flip_v", b.get_active()))
        view.append(fv)

        view.append(Gtk.Label(label="gain"))
        gain = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.0, 8.0, 0.5)
        gain.set_tooltip_text("Display brightness only — the model sees the un-gained\n"
                              "image. Use Device > exposure to give the MODEL more light.")
        gain.set_value(self.controls.gain); gain.set_size_request(110, -1)
        gain.set_draw_value(True)  # show the current number on the slider
        gain.connect("value-changed", lambda s: setattr(self.controls, "gain", s.get_value()))
        view.append(gain)

        # ---- Tracking: pipeline stages on/off + crop -------------------------
        trk = group("Tracking", grid, 1, 0)
        skel = Gtk.CheckButton(label="skeleton")
        skel.set_tooltip_text("Run hand tracking (ONNX). Off = raw frame inspection,\n"
                              "no CPU spent on inference.")
        skel.set_active(self.controls.skeleton)
        skel.connect("toggled", lambda b: setattr(self.controls, "skeleton", b.get_active()))
        trk.append(skel)

        temp = Gtk.CheckButton(label="temporal")
        temp.set_tooltip_text("Feed last frame's joints back into the keypoint net as\n"
                              "a prior (Mercury convention). Big stability win; turn off\n"
                              "to see per-frame raw performance.")
        temp.set_active(self.controls.temporal)
        temp.connect("toggled", lambda b: setattr(self.controls, "temporal", b.get_active()))
        trk.append(temp)

        und = Gtk.CheckButton(label="undistort")
        und.set_tooltip_text("Remove fisheye using the on-device calibration\n"
                             "(stereo-rectified). Required for 3D triangulation.")
        und.set_active(self.controls.undistort)
        und.connect("toggled", lambda b: setattr(self.controls, "undistort", b.get_active()))
        trk.append(und)

        kin = Gtk.CheckButton(label="kinematic")
        kin.set_active(self.controls.kinematic)
        kin.set_tooltip_text("Fit a rigid 26-DOF hand model (your measured bone\n"
                             "lengths + joint limits) to the 3D joints. Cyan\n"
                             "overlay = fitted skeleton. Kills bone stretch and\n"
                             "depth outliers; pinch/depth readouts use the fit.")
        kin.connect("toggled", lambda b: setattr(self.controls, "kinematic", b.get_active()))
        trk.append(kin)

        trk.append(Gtk.Label(label="crop"))
        crop = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.8, 2.5, 0.1)
        crop.set_tooltip_text("Size of the square box around a detected hand that the\n"
                              "keypoint net sees (x detector size). Too small clips\n"
                              "fingers; too big wastes the 128x128 crop on background.")
        crop.set_value(self.controls.crop_scale); crop.set_size_request(110, -1)
        crop.set_draw_value(True)
        crop.connect("value-changed", lambda s: setattr(self.controls, "crop_scale", s.get_value()))
        trk.append(crop)

        # ---- Device: hardware controls, sent to openleap over its stdin ------
        dev = group("Device", grid, 0, 1)
        dev.append(Gtk.Label(label="IR LEDs"))
        # all 8 combinations; bit0=left bit1=center bit2=right
        ir_masks = [0b101, 0b011, 0b110, 0b111, 0b001, 0b010, 0b100, 0b000]
        ir = Gtk.DropDown.new_from_strings([
            "2 (left+right)", "2 (left+center)", "2 (center+right)", "3 (all)",
            "1 (left)", "1 (center)", "1 (right)", "off"])
        ir.set_selected(0)  # matches openleap's left+right default
        ir.set_tooltip_text("Which IR illumination LEDs are lit. left+right lights\n"
                            "both cameras evenly at moderate power. More LEDs =\n"
                            "more range/brightness but more current draw.")
        ir.connect("notify::selected", lambda d, _:
                   self._send(f"leds {ir_masks[d.get_selected()]}"))
        dev.append(ir)

        self.ae_btn = Gtk.CheckButton(label="auto-exposure")
        self.ae_btn.set_tooltip_text("Auto-adjust exposure toward mean brightness 70.\n"
                                      "Moving the exposure slider turns this off.")
        self.ae_btn.set_active(False)  # matches openleap: manual 200 us by default
        self.ae_btn.connect("toggled", lambda b:
                            self._send(f"ae {1 if b.get_active() else 0}"))
        dev.append(self.ae_btn)

        dev.append(Gtk.Label(label="exposure µs"))
        # user-measured: all the visible change is 100-1000 us; ~1000 is near
        # full IR blowout (hand saturates to white). Slider spans the real range.
        exp = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 50, 1500, 25)
        exp.set_tooltip_text("Sensor integration time (real light for the net —\n"
                             "unlike gain, which only brightens the screen). Useful\n"
                             "range ~100-1000 us; ~1000 is near IR blowout (white hand\n"
                             "= no texture for the model). Setting this disables AE.\n"
                             "Watch the status bar: aim p95 ~200 with no CLIP; 'low'\n"
                             "means turn it up, 'CLIP' means turn it down.")
        exp.set_value(200); exp.set_size_request(170, -1)  # user-tuned default
        exp.set_draw_value(True)  # show the µs number on the slider
        # a manual exposure implies AE off (openleap does this; mirror it in UI)
        exp.connect("value-changed", lambda s: (
            self._send(f"exp {int(s.get_value())}"),
            self.ae_btn.set_active(False)))
        dev.append(exp)

        # ---- Capture: saving data to disk ------------------------------------
        cap = group("Capture", grid, 1, 1)
        rec = Gtk.CheckButton(label="record")
        rec.set_tooltip_text("Save every frame (model input PGMs + results.jsonl,\n"
                             "incl. 3D) to tracking/recordings/<timestamp>/ for\n"
                             "offline analysis. ~9 MB/s in both-eyes mode.")
        rec.connect("toggled", self._on_record)
        cap.append(rec)

        snap = Gtk.Button(label="snapshot")
        snap.set_tooltip_text("Save the current frame once: raw stereo PGM +\n"
                              "annotated PNG + skeleton/3D JSON -> tracking/snapshots/.")
        snap.connect("clicked", self._on_snapshot)
        cap.append(snap)

        cal = Gtk.Button(label="calibrate bones")
        cal.set_tooltip_text("Measure YOUR hand's bone lengths from live 3D and\n"
                             "save tracking/bone_lengths.npz (kinematic fit uses\n"
                             "them). Needs both eyes + undistort + skeleton on.\n"
                             "Hold your hand open, fingers spread, and rotate it\n"
                             "slowly so every joint is seen clearly. Click again\n"
                             "to finish early. Auto-saves when every bone is sampled.")
        cal.connect("clicked", self._on_calibrate)
        cap.append(cal)
        self._cal_btn = cal

        win.connect("close-request", self._on_close)
        win.present()

        self._start_stream()
        GLib.timeout_add(20, self._tick)  # 50 Hz sample (only redraws on new frames)

    def _start_stream(self):
        # One pkexec for everything privileged: stop the closed service (so the
        # device is free to claim) then exec the streamer. `exec` keeps the pipe
        # and signals attached to openleap.
        leap = os.path.abspath(LEAP_OPEN)
        seq = os.path.abspath(SEQ)
        cmd = ["pkexec", "sh", "-c",
               f"systemctl stop ultraleap-hand-tracking-service 2>/dev/null; "
               f"exec '{leap}' stream '{seq}'"]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0)
        except FileNotFoundError as e:
            self.status.set_text(f"failed to launch openleap: {e}")
            return
        # pump openleap's stderr: echo it (so the terminal log is unchanged) and
        # scrape the diag line for the real exposure / AE state.
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self.reader = FrameReader(self.proc)
        self.reader.start()
        self.infer = Inference(self.reader, self.controls)
        self.infer.start()

    def _on_record(self, btn):
        if btn.get_active():
            d = os.path.join(HERE, "recordings", time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(d, exist_ok=True)
            self.controls.record_dir = d
            self.controls.record = True
            print(f"recording -> {d}", file=sys.stderr)
        else:
            self.controls.record = False
            d = self.controls.record_dir
            if d:
                n = len([f for f in os.listdir(d) if f.endswith(".pgm")])
                print(f"recording stopped: {n} frames in {d}", file=sys.stderr)

    def _on_snapshot(self, _btn):
        """Export the current frame once: raw stereo PGM (full fidelity),
        annotated PNG (exactly what the viewer shows), and the skeleton/3D
        data as JSON."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        d = os.path.join(HERE, "snapshots", ts)  # one folder per snapshot
        os.makedirs(d, exist_ok=True)
        saved = []
        if self.reader:
            item = self.reader.get()
            if item:
                counter, arr = item
                p = os.path.join(d, "raw_interleaved.pgm")
                cv2.imwrite(p, arr)
                saved.append(p)
                # convenience view: same processing as the display (eye select,
                # undistort, flips, rotation, gain) but NO skeleton overlay
                c = self.controls
                cal = self.infer.cal if self.infer else None
                eyes = (0, 1) if c.eye == 2 else (c.eye,)
                imgs = [cv2.convertScaleAbs(orient(arr, c, cal, eye=e),
                                            alpha=c.gain) for e in eyes]
                clean = imgs[0]
                if len(imgs) == 2:
                    div = np.full((imgs[0].shape[0], 4), 60, np.uint8)
                    clean = np.hstack([imgs[0], div, imgs[1]])
                p = os.path.join(d, "processed_clean.png")
                cv2.imwrite(p, clean)
                saved.append(p)
        if self.infer:
            rgb, info, _ = self.infer.get()
            if rgb is not None:
                p = os.path.join(d, "annotated.png")
                cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                saved.append(p)
            if info:
                c = self.controls
                p = os.path.join(d, "info.json")
                with open(p, "w") as f:
                    json.dump({"info": info,
                               "settings": {"eye": c.eye, "rot": c.rot,
                                            "flip_h": c.flip_h, "flip_v": c.flip_v,
                                            "undistort": c.undistort,
                                            "crop": round(c.crop_scale, 2),
                                            "temporal": c.temporal}},
                              f, indent=1)
                saved.append(p)
        print(f"snapshot: {', '.join(saved) if saved else 'nothing to save yet'}",
              file=sys.stderr)
        self.status.set_text(f"snapshot saved -> snapshots/{ts}/"
                             if saved else "snapshot: no frame yet")

    # minimum confident samples on the weakest bone before an early finish is
    # allowed to actually save (below this, "finish early" just cancels).
    _BONE_MIN_SAVE = 40

    def _on_calibrate(self, _btn):
        if self._bone_cal is not None:           # second click: finish early
            self._finish_bone_cal(save=True)
            return
        if BoneCalibrator is None or self.infer is None:
            self.status.set_text("bone calibration unavailable (no kinematic module)")
            return
        c = self.controls
        # triangulation prerequisites — same gates as _triangulate()
        if not (c.skeleton and c.eye == 2 and c.undistort
                and not c.rot and not c.flip_h and not c.flip_v
                and self.infer.cal is not None and self.infer.cal.get("rectified")):
            self.status.set_text("calibrate bones needs: skeleton ON, both eyes, "
                                 "undistort ON, no rotation/flip")
            return
        self._bone_cal = BoneCalibrator()
        self.infer.bone_sampler = self._bone_cal
        self._cal_btn.set_label("finish calibration")

    def _finish_bone_cal(self, save):
        cal = self._bone_cal
        self._bone_cal = None
        if self.infer is not None:
            self.infer.bone_sampler = None
        if self._cal_btn is not None:
            self._cal_btn.set_label("calibrate bones")
        if cal is None:
            return
        if not save or cal.min_count() < self._BONE_MIN_SAVE:
            self.status.set_text(
                f"bone calibration cancelled (weakest bone only "
                f"{cal.min_count()} samples; need {self._BONE_MIN_SAVE})")
            return
        path = os.path.join(HERE, "bone_lengths.npz")
        if os.path.exists(path):  # never destroy a prior calibration
            try:
                os.replace(path, path + ".bak")
            except OSError:
                pass
        lengths = cal.save(path)
        if self.infer is not None:
            self.infer.reload_bone_lengths(lengths)
            self.controls.kinematic = True  # show the improved fit immediately
        res = cal.result()
        worst = max(range(len(res)), key=lambda i: res[i][1])  # largest MAD
        from kinematic import BONE_NAMES
        print("bone lengths (mm, median ± MAD, n):", file=sys.stderr)
        for nm, (med, mad, n) in zip(BONE_NAMES, res):
            print(f"  {nm:12s} {med:6.1f} ± {mad:4.1f}  (n={n})", file=sys.stderr)
        self.status.set_text(
            f"bone lengths saved ({cal.frames} frames) — wrist→middle "
            f"{res[8][0]:.0f}mm, index {res[4][0]:.0f}mm; noisiest "
            f"{BONE_NAMES[worst]} ±{res[worst][1]:.1f}mm. Kinematic updated.")

    def _send(self, cmd):
        """Send a device command line (leds/exp/ae) to openleap's stdin."""
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write((cmd + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    _RX_EXP = re.compile(r"exp=(\d+)us")
    _RX_MEAN = re.compile(r"mean~([\d.]+)")
    _RX_AE = re.compile(r"ae=(\d)")

    def _pump_stderr(self):
        """Read openleap's stderr line by line: scrape the diag line for the
        real exposure / AE state, and pass every line through to our own stderr
        so the terminal log is exactly as before."""
        stream = self.proc.stderr
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace")
            if "diag:" in line:
                m = self._RX_EXP.search(line)
                if m:
                    self._dev_exp = int(m.group(1))
                m = self._RX_MEAN.search(line)
                if m:
                    self._dev_mean = float(m.group(1))
                m = self._RX_AE.search(line)
                if m:
                    self._dev_ae = m.group(1) == "1"
            sys.stderr.write(line)
            sys.stderr.flush()

    def _expo_segment(self, info):
        """' | exp 240us(AE) p95 142/low' — device exposure truth plus a
        hand-exposure health tag from the frame the net actually saw."""
        if self._dev_exp is None:
            return ""
        seg = f" | exp {self._dev_exp}us({'AE' if self._dev_ae else 'man'})"
        e = info.get("expo") if info else None
        if e:
            if e["clip"] > 0.5:
                tag = "CLIP"          # too hot: shading lost to white
            elif e["p95"] < 120:
                tag = "low"           # too dim: net starved of light
            else:
                tag = "ok"
            seg += f" p95 {e['p95']:.0f}/{tag}"
        return seg

    def _tick(self):
        if self.infer is None:
            return True
        # device fps = frames received from openleap per second, regardless of
        # how many we display. Should sit at ~110-115 (the LMC's native rate);
        # a dip means the device throttled or frames are being lost upstream.
        now = GLib.get_monotonic_time()
        if now - self._dev_t > 1_000_000 and self.reader is not None:
            total, _, _ = self.reader.stats()
            self._dev_fps = (total - self._dev_prev) * 1e6 / (now - self._dev_t)
            self._dev_prev = total
            self._dev_t = now
        rgb, info, n = self.infer.get()
        if rgb is not None and n != self._last_shown_n:
            self._last_shown_n = n
            h, w, _ = rgb.shape
            tex = Gdk.MemoryTexture.new(
                w, h, Gdk.MemoryFormat.R8G8B8,
                GLib.Bytes.new(rgb.tobytes()), w * 3)
            self.pic.set_paintable(tex)
            # true inference fps (counts only new inference results)
            self._fps_n += 1
            now = GLib.get_monotonic_time()
            if now - self._fps_t > 1_000_000:
                self._fps = self._fps_n * 1e6 / (now - self._fps_t)
                self._fps_t = now; self._fps_n = 0
            ex = info.get("exists", 0.0)
            kc = info.get("kp_conf", 0.0)
            err = info.get("err", "")
            c = self.controls
            orient_str = (f"eye={('L', 'R', 'LR')[c.eye]} rot={c.rot}"
                          f"{' Hflip' if c.flip_h else ''}{' Vflip' if c.flip_v else ''}"
                          f"{' undist' if c.undistort else ''}")
            if c.skeleton:
                tri = ""
                for h in info.get("hands3d", []):
                    kin = "·kin" if h.get("kin") else ""
                    rms = f", rms {h['rms_mm']:.0f}" if "rms_mm" in h else ""
                    tri += (f" | 3D{kin}: pinch {h['pinch_mm']:.0f}mm "
                            f"depth {h['depth_mm']:.0f}mm ({h['valid']}/21{rms})")
                self.status.set_text(
                    f"device {self._dev_fps:.0f} fps | infer {self._fps:.0f} fps | "
                    f"hand_exists={ex:.2f} kp_conf={kc:.2f} | {orient_str}{tri}"
                    + self._expo_segment(info)
                    + (f" | ERR {err}" if err else ""))
            else:
                total, drops, resyncs = self.reader.stats()
                self.status.set_text(
                    f"device {self._dev_fps:.0f} fps | display {self._fps:.0f} fps | "
                    f"frames={total} drops={drops} resyncs={resyncs} | {orient_str}"
                    + self._expo_segment(info)
                    + " (skeleton off — raw)")
        elif self.proc and self.proc.poll() is not None:
            self.status.set_text("stream ended (device unplugged or auth denied)")
        # bone calibration takes over the status line (video keeps updating
        # above so the user can pose their hand) and auto-finishes when full.
        cal = self._bone_cal
        if cal is not None:
            if cal.done():
                self._finish_bone_cal(save=True)
            else:
                self.status.set_text(
                    f"calibrating bones {100 * cal.progress():.0f}% — open hand, "
                    f"spread fingers, rotate slowly (weakest bone "
                    f"{cal.min_count()}/{cal.target}, {cal.frames} frames)")
        return True

    def _on_close(self, *_):
        if self.infer:
            self.infer.stop()
        if self.reader:
            self.reader.stop()
        # openleap runs as root (pkexec), so we can't signal it directly;
        # closing the pipe gives it EOF and it stops itself + releases the device.
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                if self.proc.stdout:
                    self.proc.stdout.close()
                self.proc.terminate()
            except (PermissionError, ProcessLookupError, ValueError):
                pass
        return False


if __name__ == "__main__":
    app = Viewer()
    app.run(None)
