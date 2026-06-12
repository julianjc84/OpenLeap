#!/usr/bin/env python3
"""Pull stereo IR image frames out of a leapd bulk-transfer usbmon capture.

leapd streams images over a USB BULK IN endpoint. Each completed bulk URB
('C' callback, xfer_type 3, direction IN) carries a chunk of pixel data.
We concatenate those chunks in capture order into one byte stream, find the
frame period, de-interleave each frame into left/right 8-bit images
(YUY2-style L/R interleave per the LeapUVC manual), and save a few PNGs.

Frame geometry is unknown up front, so we test candidate interleaved widths
and pick the one whose row-autocorrelation is strongest (real images have
strong horizontal self-similarity at the true stride).

Input: a classic-pcap usbmon file (convert pcapng first:
  editcap -F pcap in.pcapng out.pcap).
"""

import struct
import sys
from pathlib import Path

import numpy as np

USBMON_HDR = 64
OUTDIR = Path(__file__).resolve().parent / "results" / "frames"


def read_pcap_records(path):
    data = Path(path).read_bytes()
    (magic,) = struct.unpack_from("<I", data, 0)
    if magic != 0xA1B2C3D4:
        raise SystemExit(f"need classic LE pcap; got magic 0x{magic:08x}")
    off, n = 24, len(data)
    while off + 16 <= n:
        _, _, incl, _ = struct.unpack_from("<IIII", data, off)
        off += 16
        if off + incl > n:
            break
        yield data[off : off + incl]
        off += incl


def bulk_in_stream(path):
    chunks = []
    total = 0
    for rec in read_pcap_records(path):
        if len(rec) < USBMON_HDR:
            continue
        typ = chr(rec[8])
        xfer = rec[9]
        epnum = rec[10]
        (len_cap,) = struct.unpack_from("<I", rec, 36)
        # bulk (3), IN (epnum bit7), completion carries the data
        if xfer != 3 or not (epnum & 0x80) or typ != "C" or len_cap == 0:
            continue
        payload = rec[USBMON_HDR : USBMON_HDR + len_cap]
        chunks.append(payload)
        total += len(payload)
    return b"".join(chunks), total


def score_width(buf, width, rows=240, samples=12):
    """Mean row-to-row correlation at a candidate interleaved width."""
    frame_bytes = width * rows
    if buf.size < frame_bytes * 2:
        return -1
    scores = []
    for s in range(samples):
        start = (s + 1) * frame_bytes
        if start + frame_bytes > buf.size:
            break
        img = buf[start : start + frame_bytes].reshape(rows, width).astype(np.float32)
        # correlation between adjacent rows; high for coherent images
        a, b = img[:-1].ravel(), img[1:].ravel()
        if a.std() < 1e-3 or b.std() < 1e-3:
            continue
        scores.append(float(np.corrcoef(a, b)[0, 1]))
    return sum(scores) / len(scores) if scores else -1


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/leap_full_classic.pcap"
    print(f"reading bulk-IN stream from {src} ...")
    raw, total = bulk_in_stream(src)
    buf = np.frombuffer(raw, dtype=np.uint8)
    print(f"bulk image bytes: {total:,} ({total/1e6:.1f} MB)")

    # Candidate geometries. Interleaved width = 2 * sensor width (L/R packed).
    # leapd tracking modes are usually 640x240 or 640x480 (manual lists both,
    # plus 752-wide variants).
    candidates = []
    for sw, h in [(640, 240), (640, 480), (640, 120), (752, 240), (752, 480), (752, 120)]:
        candidates.append((sw, h, sw * 2, h))
    best = None
    print("\n  sensorW  h   interleavedW   row-corr")
    for sw, h, iw, rows in candidates:
        sc = score_width(buf, iw, rows)
        print(f"   {sw:>4}  {h:>4}      {iw:>5}       {sc:+.3f}")
        if best is None or sc > best[0]:
            best = (sc, sw, h, iw, rows)
    sc, sw, h, iw, rows = best
    print(f"\nbest geometry: sensor {sw}x{h}, interleaved {iw}x{rows}, corr {sc:+.3f}")

    frame_bytes = iw * rows
    nframes = buf.size // frame_bytes
    print(f"frames available: {nframes}")
    if nframes < 3:
        raise SystemExit("not enough data for the chosen geometry")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        cv2 = None

    # grab a few frames spread through the capture (skip the first ones -
    # exposure is still converging at stream start)
    picks = [int(nframes * f) for f in (0.3, 0.5, 0.7, 0.85)]
    for i, fidx in enumerate(picks):
        start = fidx * frame_bytes
        inter = buf[start : start + frame_bytes].reshape(rows, iw)
        left = inter[:, 0::2]
        right = inter[:, 1::2]
        sbs = np.hstack([left, right])
        name = OUTDIR / f"frame_{i}_idx{fidx}"
        if cv2 is not None:
            cv2.imwrite(str(name) + "_left.png", left)
            cv2.imwrite(str(name) + "_right.png", right)
            cv2.imwrite(str(name) + "_sidebyside.png", sbs)
        else:
            # raw fallback
            (Path(str(name) + "_left.gray")).write_bytes(left.tobytes())
        print(f"  saved {name.name}_* "
              f"(L mean {left.mean():.0f}, R mean {right.mean():.0f})")

    print(f"\nframes written to {OUTDIR}/")


if __name__ == "__main__":
    main()
