#!/usr/bin/env python3
"""Extract leapd's control-transfer init/unlock sequence from a usbmon pcap.

Parses the Linux usbmon-mmapped capture directly (no tshark field-name
fragility). For every control SUBMIT that goes host->device (a SET_CUR /
vendor write), emits the setup packet + data payload in capture order.

usbmon binary header layout (per Documentation/usb/usbmon.rst), little-endian:
  off 0  u64 id
  off 8  u8  type      'S'=submit 'C'=callback 'E'=error
  off 9  u8  xfer_type 0=iso 1=intr 2=control 3=bulk
  off 10 u8  epnum     bit7 = direction (0x80 = IN)
  off 11 u8  devnum
  off 12 u16 busnum
  off 14 u8  flag_setup
  off 15 u8  flag_data
  off 16 u64 ts_sec
  off 24 u32 ts_usec
  off 28 s32 status
  off 32 u32 length     (urb length)
  off 36 u32 len_cap    (bytes actually captured after the 64-byte header)
  off 40 [8] setup      (only valid for control submit when flag_setup==0)
  off 48 ...            (interval/start_frame/xfer_flags/ndesc)
  off 64 payload[len_cap]

Setup (USB spec): bmRequestType(1) bRequest(1) wValue(2,LE) wIndex(2,LE) wLength(2,LE)

Output: prints a human table + writes leap_init_sequence.json next to this file.
"""

import json
import struct
import sys
from pathlib import Path

PCAP_MAGIC_LE = 0xA1B2C3D4
USBMON_HDR = 64


def read_pcap_records(path):
    data = Path(path).read_bytes()
    magic, = struct.unpack_from("<I", data, 0)
    if magic != PCAP_MAGIC_LE:
        # try the other endianness / nanosecond magic; we only need LE micros here
        raise SystemExit(f"unexpected pcap magic 0x{magic:08x} (expected LE 0xa1b2c3d4)")
    # global header is 24 bytes
    off = 24
    n = len(data)
    while off + 16 <= n:
        ts_sec, ts_usec, incl, orig = struct.unpack_from("<IIII", data, off)
        off += 16
        if off + incl > n:
            break
        yield data[off : off + incl]
        off += incl


def parse_usbmon(rec):
    if len(rec) < USBMON_HDR:
        return None
    typ = rec[8]
    xfer = rec[9]
    epnum = rec[10]
    devnum = rec[11]
    busnum, = struct.unpack_from("<H", rec, 12)
    flag_setup = rec[14]
    length, = struct.unpack_from("<I", rec, 32)
    len_cap, = struct.unpack_from("<I", rec, 36)
    setup = rec[40:48]
    payload = rec[64 : 64 + len_cap]
    return dict(typ=chr(typ), xfer=xfer, epnum=epnum, devnum=devnum,
                busnum=busnum, flag_setup=flag_setup, length=length,
                len_cap=len_cap, setup=setup, payload=payload)


def main():
    pcap = sys.argv[1] if len(sys.argv) > 1 else "/tmp/leap_control_only.pcap"
    out = []
    seq = 0
    for rec in read_pcap_records(pcap):
        p = parse_usbmon(rec)
        if not p:
            continue
        # control transfers only, submit phase, with a valid setup packet
        if p["xfer"] != 2 or p["typ"] != "S" or p["flag_setup"] != 0:
            continue
        bmRequestType, bRequest, wValue, wIndex, wLength = struct.unpack("<BBHHH", p["setup"])
        host_to_dev = (bmRequestType & 0x80) == 0
        if not host_to_dev:
            continue  # OUT (writes) only; these carry the unlock/config
        data = p["payload"][:wLength] if wLength else b""
        seq += 1
        entry = dict(
            seq=seq,
            dev=p["devnum"],
            bus=p["busnum"],
            bmRequestType=bmRequestType,
            bRequest=bRequest,
            # UVC: wValue hi = control selector, wIndex hi = unit/entity id, lo = interface
            selector=(wValue >> 8) & 0xFF,
            wValue=wValue,
            unit=(wIndex >> 8) & 0xFF,
            interface=wIndex & 0xFF,
            wIndex=wIndex,
            wLength=wLength,
            data=data.hex(),
        )
        out.append(entry)

    # report
    print(f"{'seq':>3} {'dev':>3} {'bmReq':>5} {'bReq':>4} {'sel':>4} "
          f"{'unit':>4} {'if':>3} {'len':>4}  data")
    for e in out:
        print(f"{e['seq']:>3} {e['dev']:>3}  0x{e['bmRequestType']:02x} "
              f"0x{e['bRequest']:02x}  0x{e['selector']:02x} "
              f"{e['unit']:>4} {e['interface']:>3} {e['wLength']:>4}  {e['data']}")

    # group: which devices, units, selectors appear
    devs = {}
    for e in out:
        devs.setdefault(e["dev"], 0)
        devs[e["dev"]] += 1
    print(f"\ncontrol-OUT writes: {len(out)} total; by device: {devs}")
    units = sorted({(e['unit'], e['selector']) for e in out})
    print(f"distinct (unit, selector) pairs written: {units}")

    dst = Path(__file__).resolve().parent / "leap_init_sequence.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
