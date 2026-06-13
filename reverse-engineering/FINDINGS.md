# Phase 0 findings — can we drive the LMC dev-kit via open UVC?

Date: 2026-06-12. Device: original Leap Motion Controller, USB `f182:0003`,
self-IDs as "Leap Dev Kit", **firmware 1.7.0** (per `leapctl`; USB
`bcdDevice` reads 0000), no serial in USB descriptor.

## ✅ PHASE 0 COMPLETE (2026-06-12): our open driver streams the device.

`leap-open bringup` — a ~300-line Rust/libusb tool with zero leapd —
brought up the locked firmware-1.7.0 device from scratch and streamed it:
**358/358 control transfers replayed OK, 207 MB / 674 frames in 6 s, 0
timeouts (~112 fps, matching the 115 fps spec).** Frames decode at the
expected 640×240×2 geometry and show real IR imagery. The closed daemon is
no longer required for raw capture. See `leap-open/` and the milestone
note at the bottom of this file. The sections below are the investigation
trail that got us here.

## (historical) Result: PARTIAL PASS — hardware is fully accessible, but
## streaming is gated behind a firmware unlock.

| Test | Result |
|---|---|
| Device discovery (`/dev/videoN`, vendor f182) | ✅ enumerates as `/dev/video4`, card "Leap Dev Kit" |
| V4L2 open + format negotiation | ✅ kernel uvcvideo binds cleanly |
| Mode enumeration | ✅ **all 6 modes present, exactly matching LeapUVC manual** |
| Raw frame streaming | ❌ `select() timeout`, no frames in 180 s, IR LEDs never lit |

Enumerated modes (`v4l2-ctl --list-formats-ext`, saved in
`results/v4l2-metadata.txt`), pixel format YUYV (the interleaved stereo
the manual describes):

- 640×480 @ 57.5 fps
- 640×240 @ **115 fps**   ← Gemini's normal operating mode
- 640×120 @ 214 fps
- 752×480 @ 50 fps
- 752×240 @ 100 fps
- 752×120 @ 190 fps

## Root cause: firmware 1.7.0 needs an authentication/unlock handshake

LeapUVC-Manual.pdf, page 1, verbatim:

> NOTE: this document applies to firmware version **1.7.1 or later**.
> Previous versions of the firmware **required authentication to use**.

Our dev-kit is **1.7.0** — one revision below the cutoff. The device
enumerates as standard UVC and accepts format negotiation, but delivers
zero frames until it receives a proprietary unlock sequence. Confirmed by:
`select()` timing out for 180 s straight, and the user observing the IR
LEDs never illuminate (the device "stays cold until the camera turns on" —
it never turned on).

`leapuvc/LeapUVC Firmware Tools/UnlockDevice.exe` is the (Windows) tool
that sends this handshake. A "Device Restoration Tools" folder
(`LM_FirmwareReset_Win.exe`, `RestoreDevice.exe`) sits alongside it —
suggests the unlock may involve a firmware-state change, handle with care.

## The chicken-and-egg

- Service **running**: `leapd` unlocks the device (this is how niri's leap
  gestures work today) but then claims it exclusively — UVC clients can't
  open it.
- Service **stopped**: device is free for UVC, but **locked** — no frames.

The open path must perform the unlock itself.

## Next step (Phase 0b): capture leapd's unlock, replay via libusb

This is exactly the OpenLeap methodology — and its tooling
(`OpenLeap/make_leap_usbinit.sh` + `usb_c.lua`) already turns a USB
capture into replayable libusb init code.

Plan:
1. `pkexec modprobe usbmon`
2. Start a `tshark`/`dumpcap` capture on the usbmon bus for `f182:0003`.
3. Stop, then start `ultraleap-hand-tracking-service` so leapd performs
   its init+unlock against the device while we capture.
4. Stop the service, isolate the control transfers (setup `bRequest==1`,
   the vendor unlock), feed the pcap through `make_leap_usbinit.sh` to
   generate `leap_libusb_init.c.inc`.
5. Build a tiny libusb program (model: `OpenLeap/low-level-leap.c`) that
   sends that sequence, then re-run `probe.py` — if frames + LEDs come on,
   Phase 0 is fully green and Phase 1 (Rust capture crate) can proceed.

Caveat: OpenLeap's original captures were 2013-era; our 1.7.0 unlock may
differ, which is exactly why we capture our *own* device's sequence rather
than reusing theirs.

Fallback if capture-replay fails: run `UnlockDevice.exe` under Wine with
USB passthrough and test whether the unlock persists across the
service-stop (i.e. whether the device stays unlocked until physically
replugged).

## Phase 0b attempt (2026-06-12): captured leapd's init, but power-contaminated

Captured the USB bus (`usbmon3`, via `dumpcap`) across a service
stop→start cycle while the device was connected, to record leapd bringing
the device up. Capture preserved at
`captures/leap_unlock_2026-06-12.pcap` (1995 packets, 193 KB).

What we learned about **how leapd drives the device** (this is new and
useful regardless of the unlock specifics):

- **Image transport is USB BULK**, not isochronous/V4L2. 1600 of 1995
  packets were bulk transfers (transfer_type 0x03). So leapd detaches the
  kernel `uvcvideo` driver and talks libusb directly — it does NOT go
  through `/dev/videoN`. (That's also why a UVC client can't open the
  device while the service runs.)
- **Configuration is UVC class control writes** (`bmRequestType 0x23` =
  host→device/class/interface, i.e. SET_CUR on the video-control
  interface). 17 distinct writes per init attempt. Example setup packet
  (frame 17): `23 03 02 00 05 00 00 00` → bRequest 0x03, wValue 0x0002,
  wIndex 0x0005 — a vendor use of the UVC control channel, consistent
  with the "property knocking" mechanism the manual/leapuvc.py describe.
- **28 interrupt transfers** (status endpoint).

**Problem: the capture is power-contaminated.** The user observed the
device "dropping out partly due to power." Confirmed in the pcap: **98
GET_DESCRIPTOR(DEVICE) reads in 28 s** — the device re-enumerated dozens
of times (a clean run shows 1–2). leapd's init handshake appears ~7 times
in regular frame-number groups (207.., 469.., 734.., 968.., 1232..,
1468.., 1728..) — leapd retrying after each dropout. So we can't cleanly
isolate "the unlock" from re-init noise in this capture.

### POWER is now the gating blocker, not the protocol

The original LMC pulls significant current (3 high-power 850 nm IR LEDs).
On this machine's USB port it browns out and re-enumerates. Before the
open path can work — and arguably for stable Gemini use too — the device
needs solid power:

- a **powered USB hub**, or
- a **direct rear-panel (motherboard) USB port**, not a front-panel
  header or an unpowered hub,
- a known-good USB cable (the LMC's micro-USB is a common failure point).

A stable device is a prerequisite for both a clean unlock capture AND for
feeding usable frames to Mercury, so this comes first.

### Revised next step (Phase 0b, take 2 — once power is stable)

1. Fix power (powered hub / rear port / good cable).
2. Recapture with a *fresh plug*: service stopped, `dumpcap -i usbmon3`
   running, THEN physically plug the device, THEN start the service —
   so the very first init against a locked, freshly-enumerated device is
   recorded once, cleanly, with no dropout retries.
3. Extract the control-write sequence (payloads via `tshark -x`; the
   OUT-data stage needs the clean capture to read), feed through
   `OpenLeap/make_leap_usbinit.sh` → `leap_libusb_init.c.inc`.
4. Build a libusb replay (model `OpenLeap/low-level-leap.c`) OR — now that
   we know leapd uses **bulk + libusb directly** — prototype the whole
   capture path in Rust against libusb (rusb crate), skipping V4L2
   entirely. This actually simplifies Phase 1: the Rust capture crate
   talks libusb bulk, matching how leapd itself works.
5. Re-run `probe.py` (or the Rust equivalent) — frames + IR LEDs = green.

Open question for take-2: does stopping leapd re-lock the device, or does
the unlock persist until physical replug? The fresh-plug capture answers
it (and is the safe assumption either way).

## Phase 0b take-2 (2026-06-12): CLEAN-ish capture + decoded the protocol

Recaptured via fresh replug (service left running; user physically
unplugged → replugged so the already-running leapd performed its one-shot
bring-up against a freshly-enumerated, locked device). 45 s, **127,361
packets, 920 MB** — the device **streamed this time** (the bulk is real
stereo IR image data). Still some power churn (482 descriptor reads, and
the device re-enumerated from USB addr 55 → 56), but it recovered and
streamed, which the first capture never did.

Decoded leapd's control traffic directly with a hand-written usbmon parser
(`extract_init.py` — tshark's `usb.setup.*` fields proved unreliable here;
parsing the 64-byte usbmon header is robust). Full decoded sequence saved
to `leap_init_sequence.json` (374 control-OUT writes).

### The mechanism: two UVC vendor Extension Units, register-poke protocol

leapd does NOT send a single "unlock magic packet." It brings the device
up by writing **UVC SET_CUR (bRequest 0x01) to two vendor Extension
Units** on the VideoControl interface:

- **Extension Unit 2**, selector **0x0B** — a DATA/value register
- **Extension Unit 5**, selectors **0x03, 0x04, 0x07, 0x08, 0x0A** — an
  address/index + control register bank

The fresh-plug bring-up (USB addr 55) is **52 writes, all paired**:
write a value to (unit 2, sel 0x0B), then an index to (unit 5, sel 0x04),
repeat. The value register ramps `0x4d → 0x57 → 0x179 → 0x1d1 → 0x1ff`
then plateaus at `0x1ff` (511) while the index register climbs
`0x10…0x17` and settles — this reads like an **auto-exposure / sensor
register convergence loop**, i.e. leapd tuning the Aptina MT9V024 sensors,
not a cryptographic unlock. After bring-up the device re-enumerated (55→56)
and leapd kept driving the same (unit2,unit5) registers throughout
streaming (308 more writes on addr 56).

This maps exactly onto `leapuvc.py`, which pokes the same hardware through
OpenCV's `CAP_PROP_SHARPNESS`/`SATURATION`/`ZOOM`/`GAIN`/`CONTRAST` — those
are just the kernel-UVC names for these same Extension Unit selectors. So
the open path is: **drive Extension Units 2 & 5 with this register
sequence over libusb (or via the UVC controls), and the device streams.**

### Open questions (need a power-stable capture to resolve cleanly)

1. **Is the 55→56 re-enumeration intentional or power churn?** Some Leap
   firmwares re-enumerate after init (boot in limited mode → unlock →
   re-enumerate as full device — could explain the user's "~30 s to boot").
   But this capture still had power instability, so it's ambiguous. A
   well-powered capture settles it.
2. **Is there a true auth handshake for FW 1.7.0, or does configuring the
   extension units suffice?** The captured sequence looks like sensor
   tuning, not auth — suggesting 1.7.0 may simply need the extension-unit
   bring-up, no secret. To be confirmed by actually replaying it.

### Concrete next actions (still gated on stable power)

- Fix power (powered hub / rear port / good cable) and recapture once,
  cleanly, to resolve the two questions above.
- Prototype the replay: open the device with `rusb` (libusb), send the
  `leap_init_sequence.json` writes to Extension Units 2 & 5, then read the
  bulk endpoint — if frames + LEDs come up, the open capture path is real.
- Decode the bulk image framing (the 920 MB capture has thousands of real
  frames) to recover left/right IR images for feeding Mercury in Phase 2.

## Tooling / artifacts added this session

- `reverse-engineering/probe.py` — UVC capture probe (V4L2 path; times out pre-unlock)
- `reverse-engineering/extract_init.py` — usbmon→sequence decoder (no tshark dependency)
- `reverse-engineering/usb_c_patched.lua` — OpenLeap lua with the renamed-field fix
  (kept for reference; the Python parser superseded it)
- `reverse-engineering/leap_init_sequence.json` — 374 decoded control-OUT writes
- `reverse-engineering/captures/leap_init_clean_2026-06-12.pcap{,ng}` — the clean
  control-only capture (small)
- `reverse-engineering/captures/leap_unlock_2026-06-12.pcap` — first (power-contaminated)
  capture
- `reverse-engineering/extract_frames.py` — bulk-stream → IR frame decoder
- `reverse-engineering/results/frames/` — **4 real stereo IR frames** decoded from the
  open path (left/right/side-by-side PNGs). The 920 MB full capture was
  discarded after extraction (too big for the project drive).

### Image framing decoded (from the 920 MB capture, now discarded)

- leapd's tracking-mode geometry is **640×240 per eye**, transmitted
  **interleaved L/R → 1280×240, 8-bit grayscale** (YUY2-style packing per
  the LeapUVC manual). Confirmed by row-autocorrelation: +0.980 at this
  geometry vs +0.54 for the 752-wide candidates.
- 2,958 frames were present in the 45 s capture; the saved 4 clearly show
  an IR-lit hand/forearm against a dark room, with correct stereo parallax
  and the expected fisheye vignette.
- Known limitation for a real decoder: frames must be **synced on the
  12-byte embedded metadata line** (bottom-right of each frame), not a
  fixed byte stride — the saved montages show slight diagonal tearing from
  fixed-stride concatenation across occasional partial bulk chunks. Phase 1
  capture crate should frame-sync properly.

## Artifacts in `results/` and `captures/`

- `v4l2-metadata.txt` — full `v4l2-ctl --all` + mode list
- (`probe.py` would also write `left.png`/`right.png`/`calibration.json`
  once streaming works — not yet produced)

## Tooling notes

- Probe + deps live in `OpenLeap/.venv` (opencv, numpy,
  scipy). Run: `.venv/bin/python reverse-engineering/probe.py`.
- Sudo here needs the polkit GUI (`pkexec ...`), not passwordless sudo.
- Stop/start the service with
  `pkexec systemctl {stop,start} ultraleap-hand-tracking-service`.

## MILESTONE: leap-open driver (Phase 0 → Phase 1 bridge)

`OpenLeap/leap-open/` — Rust + `rusb` (libusb). Two commands:

- `info` — dumps the descriptor tree, finds endpoints, proves we can
  detach `uvcvideo` and claim the device. Discovered:
  VideoControl = iface 0 (interrupt EP 0x82), VideoStreaming = iface 1
  (**bulk IN EP 0x83**, 512-byte max packet), device max power **1000 mA**
  (explains the brownouts — a hungry device for USB 2).
- `bringup [seqfile] [secs]` — claims ifaces 0+1, replays the captured
  bring-up sequence (`bringup_seq.txt`, 358 control transfers = leapd's
  extension-unit config incl. the UVC VS_PROBE/VS_COMMIT stream-start),
  then reads bulk EP 0x83 and saves de-interleaved frames as PGM.

Result of the first live run: full success (see top of file). This
confirms the earlier conclusion — FW 1.7.0 has **no cryptographic unlock**;
it just needs leapd's extension-unit bring-up + standard UVC probe/commit,
which we now reproduce ourselves.

### Known refinements (next)

1. **Frame sync**: the bulk reader currently concatenates at a fixed
   307200-byte stride, so saved frames show diagonal shearing across
   partial-chunk boundaries. Sync on the 12-byte embedded metadata line
   (bottom-right of each frame) for clean frames.
2. **Auto-exposure**: bring-up replays leapd's converged AGC end-state
   (a frozen exposure). For a real driver, implement our own AGC by
   writing the exposure/gain extension-unit registers (unit 2 sel 0x0b =
   value, unit 5 = index) based on frame brightness.
3. **Robustness**: handle the bring-up without a pre-recorded capture —
   distill the 358-transfer replay down to the minimal required set
   (drop the 276 redundant AGC writes), and parameterize exposure.

### Path from here

- **Phase 1**: turn `bringup` into a clean capture library — frame-synced,
  exposure-controlled, exposing a `next_frame() -> (left, right)` API.
  This is the open analogue of leapd for raw images.
- **Phase 2**: feed those frames into Mercury
  (`../monado` + `../hand-tracking-models`) and evaluate skeleton quality
  on the desk-mounted, palm-up viewpoint.
- Then a thin shim emits `LeapHandData` and niri's recognizer runs
  unchanged on the fully-open stack.

## UPDATE (2026-06-12): frame-sync + auto-exposure DONE

Both Phase-1 refinements landed and were validated live.

**Bulk framing decoded.** Each bulk read returns one UVC payload of
**16380 bytes = 12-byte UVC header + 16368 pixel bytes**. Header:
`byte0=0x0c` (bHeaderLength 12), `byte1=bmHeaderInfo` (bit0 FrameID,
bit1 EndOfFrame, bit7 EndOfHeader), then a presentation timestamp. A frame
= 18×16368 + 1×12576 pixel bytes = exactly 307200, spread over ~19
payloads, the last carrying the EOF bit. (My earlier fixed-307200-stride
guess caused the diagonal shear — wrong because it ignored the per-payload
headers.)

**Frame-sync (fixed):** strip each payload's `bHeaderLength` header,
concatenate pixels, cut the frame when the EOF bit is set. Result:
623 geometrically perfect, shear-free frames in 8 s (~78–86 fps after
dropped-payload losses; raw payload rate ~115 fps). Proof image:
`results/frames/leap-open_synced_hand.png` — a crisp stereo IR hand, five
fingers, clean parallax, no tearing.

**Auto-exposure (working, register confirmed):** closed loop writing
exposure (µs) via UVC SET_CUR, **extension unit 2, selector 0x0b**
(wValue 0x0b00, wIndex 0x0200, 4-byte LE), multiplicative control toward a
target mean of 70, clamped [10, 8000] µs. Live trace confirmed the
register controls exposure: started 400µs @ mean 100 → walked down to
~92µs to hit target; when the hand left and the scene darkened (mean 16),
ramped up toward the 8000µs ceiling. `unit 2 sel 0x0b = exposure` is now a
verified fact, not a hypothesis.

Minor follow-ups (not blocking): AE overshoots when the scene changes
abruptly (hand in/out) — a slew-rate limit or smaller step would damp it;
and `save_frames` should pick the best-exposed frame (nearest target mean)
rather than fixed percentiles, so the demo frame isn't a dark one.

**Phase 1 capture path is essentially complete**: bring-up + frame-synced,
exposure-controlled streaming over pure libusb, no leapd. Next is wrapping
this as a `next_frame() -> (left,right)` library and pointing it at Mercury
(Phase 2).
