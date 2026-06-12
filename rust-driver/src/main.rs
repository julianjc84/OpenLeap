//! `openleap` binary — a thin wrapper around the `openleap` library (`lib.rs`).
//!
//! The library holds all the device logic (bring-up, stereo IR streaming,
//! exposure/LED control, calibration read) and additionally exposes a C ABI in
//! `src/ffi.rs` (`leap_open_*`) for embedding — e.g. the Monado `leap_open`
//! driver links `libopenleap.a` and drives the device through that ABI.

fn main() {
    openleap::cli_main();
}
