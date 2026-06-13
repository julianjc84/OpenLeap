# OpenLeap — Licensing posture

Why OpenLeap can exist and be published openly, and the lines it must not
cross. **Not legal advice** — this is the project's documented reasoning under
the Ultraleap Tracking SDK Agreement (English law). The full clause-by-clause
analysis of the closed stack lives in a private companion document, maintained
separately and not shipped in this repo.

## The short version

**OpenLeap (our USB driver + Monado's Mercury engine) does not breach the
Ultraleap EULA.** It uses none of Ultraleap's SDK, software, or trained
weights; it reaches the device through an independently reverse-engineered USB
protocol; and — because it never uses their software — the SDK agreement
arguably does not govern this path at all.

## What OpenLeap does and does not touch

| | Source | License status |
|---|---|---|
| USB bring-up + stereo capture | **ours** (`rust-driver/`, libusb) — protocol decoded from usbmon capture | clean-room; ours to publish |
| Hand skeleton | **Monado Mercury** (`ht` driver) | open (BSL engine) |
| Skeleton models | `hand-tracking-models/` (separate clone) | CC-BY-SA 4.0 (Monado/Collabora) — attribute, never vendor in |
| Ultraleap SDK / `libLeapC.so` / `leapd` / `.ldat` weights | **not used, not linked, not bundled** | their closed materials — stay out |

## The two EULA clauses that could bite — and why they don't

**§3.1.6 — "no reverse engineering."** This binds reverse engineering of
*their Software* (SDK, `libLeapC`, `leapd`, the service). We did reverse-
engineer — but the **device's USB protocol**, by observing `leapd`'s traffic
with usbmon (see `reverse-engineering/FINDINGS.md`), never decompiling their
binaries. That targets hardware/wire behaviour, not their software internals,
and the clause itself carves out **"except as applicable law permits"**: UK/EU
statutory **interoperability exceptions** (which a contract cannot override)
cover exactly this — analysis done to build an independent, interoperable
client.

**§3.1.5 + §18.1 — "no using the SDK or its Image API to build competing
hand-tracking software."** This is the clause that matters for an open
replacement engine, and OpenLeap **deliberately sidesteps it**: pixels come
from *our own libusb driver*, never their SDK, never LeapUVC, never their Image
API. Mercury is fed by *our* frames.

> ⚠️ **The trap to avoid:** LeapUVC is *also* Ultraleap SDK material. Feeding
> Mercury (or any tracker) from LeapUVC would arguably breach §18.1. OpenLeap
> must always source frames from its own driver, never LeapUVC.

## Why the SDK agreement may not even apply here

The SDK agreement binds **licensees of the SDK**. Running OpenLeap doesn't use
their software and doesn't require accepting their EULA (`leapctl eula`). You
only become bound by accepting the EULA to run the *official* stack — the
pure-OpenLeap path exercises a license it never needed.

## Lines OpenLeap must not cross

- **Never bundle or re-host Ultraleap's closed materials.** Not `libLeapC.so`,
  not `leapd`, not the service `.deb`s, not the `.ldat` weights. Private
  archival for personal preservation (in a private archive kept outside this
  repo) is a separate, grey abandonware matter — but **nothing closed ships in
  this repo or in any public release of it.**
- **Don't link OpenLeap into a GPL binary you then distribute** if that binary
  also links Ultraleap's SDK (§3.1.8 anti-copyleft + GPL friction). OpenLeap
  itself is SDK-free, so this only matters if a downstream consumer mixes both
  — e.g. niri. That's why niri's `leap` cargo feature stays **off by default**
  and niri binaries with it linked are never distributed.
- **The LeapC shim (if built) must be a clean reimplementation** of the public
  `LeapC` API surface, backed by Mercury — not a copy of, or a wrapper around,
  Ultraleap's `libLeapC.so`. Reimplementing a published API for interop is the
  same interoperability-exception territory as the USB work.

## This project's own code

Open (license TBD before first publish). The repo contains **none** of
Ultraleap's files. The protocol knowledge was obtained by observing the
device's own USB traffic — a clean-room path that never touched their SDK.
