//! C ABI for embedding the Leap Open capture path (the Monado `leap_open`
//! driver links `libopenleap.a` and calls these). All bring-up / streaming /
//! control logic is shared with the `openleap` binary — this only wraps it in
//! `extern "C"`.
//!
//! Lifecycle: `leap_open_open` -> (optional) `leap_open_read_calib` ->
//! `leap_open_start` -> `leap_open_next_frame` in a loop -> `leap_open_close`.
//! Control setters (`exposure`/`ae`/`leds`) are safe to call any time after
//! open. Every function is `unsafe` because it dereferences caller-owned raw
//! pointers; the markers are invisible to C callers.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use rusb::{Context, DeviceHandle};

use crate::{
    drain_usb, open_claimed, parse_seq, read_calibration, replay_seq, set_exposure, set_leds,
    Counters, DiagPrev, Mailbox, StreamState, VC_IFACE, VS_IFACE,
};

/// On-device calibration size: PU addresses 100..256 -> 156 bytes (matches
/// `read_calibration` and the leapuvc struct layout).
const CALIB_BYTES: usize = 156;

/// Opaque handle passed back and forth across the C boundary. Owns the
/// brought-up device + the shared control atomics, and (once started) the RT
/// drain thread + frame mailbox.
pub struct LeapOpen {
    handle: Arc<DeviceHandle<Context>>,
    exposure: Arc<AtomicU32>,
    ae_on: Arc<AtomicBool>,
    led_mask: Arc<AtomicU32>,
    stream: Option<Streaming>,
}

struct Streaming {
    mailbox: Arc<Mailbox>,
    drain: Option<JoinHandle<()>>,
    st: StreamState,
}

/// # Safety
/// `p` must be a valid NUL-terminated C string or null.
unsafe fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok().map(str::to_owned)
}

/// Open + claim the Leap and replay leapd's bring-up. `seq_path` = path to
/// `bringup_seq.txt`. Returns null on failure (device missing, bad seq path,
/// claim refused — e.g. not root / Ultraleap service still holding it).
///
/// # Safety
/// `seq_path` must be a valid NUL-terminated C string or null.
#[no_mangle]
pub unsafe extern "C" fn leap_open_open(seq_path: *const c_char) -> *mut LeapOpen {
    let path = match cstr(seq_path) {
        Some(p) => p,
        None => {
            eprintln!("leap_open_open: null/invalid seq_path");
            return std::ptr::null_mut();
        }
    };
    let seq = match parse_seq(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("leap_open_open: cannot read '{path}': {e}");
            return std::ptr::null_mut();
        }
    };
    let handle = match open_claimed() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("leap_open_open: {e}");
            return std::ptr::null_mut();
        }
    };
    replay_seq(&handle, &seq);
    Box::into_raw(Box::new(LeapOpen {
        handle: Arc::new(handle),
        exposure: Arc::new(AtomicU32::new(8000)), // bright default (Mercury sees raw frames)
        ae_on: Arc::new(AtomicBool::new(false)),
        led_mask: Arc::new(AtomicU32::new(0b101)), // left+right
        stream: None,
    }))
}

/// Read the 156-byte on-device stereo calibration into `out` (must hold at
/// least `CALIB_BYTES`). Must be called BEFORE `leap_open_start` (it does
/// control transfers; the drain thread owns the bulk endpoint once streaming).
/// Returns 0 on success, -1 on bad args / short read, -2 if already streaming.
///
/// # Safety
/// `h` must come from `leap_open_open`; `out` must point to >= `CALIB_BYTES`.
#[no_mangle]
pub unsafe extern "C" fn leap_open_read_calib(h: *mut LeapOpen, out: *mut u8) -> i32 {
    let lo = match h.as_ref() {
        Some(l) => l,
        None => return -1,
    };
    if lo.stream.is_some() {
        return -2;
    }
    let calib = read_calibration(&lo.handle);
    if calib.len() != CALIB_BYTES || out.is_null() {
        return -1;
    }
    std::ptr::copy_nonoverlapping(calib.as_ptr(), out, CALIB_BYTES);
    0
}

/// Begin streaming: apply current LED/exposure and spawn the RT drain thread.
/// Idempotent. Returns 0 on success, -1 on bad handle.
///
/// # Safety
/// `h` must come from `leap_open_open`.
#[no_mangle]
pub unsafe extern "C" fn leap_open_start(h: *mut LeapOpen) -> i32 {
    let lo = match h.as_mut() {
        Some(l) => l,
        None => return -1,
    };
    if lo.stream.is_some() {
        return 0;
    }

    let applied_leds = lo.led_mask.load(Ordering::Relaxed) as u8;
    set_leds(&lo.handle, applied_leds);
    set_exposure(&lo.handle, lo.exposure.load(Ordering::Relaxed));

    let mailbox = Arc::new(Mailbox::default());
    let counters = Arc::new(Counters::default());
    let drain = {
        let handle = Arc::clone(&lo.handle);
        let mailbox = Arc::clone(&mailbox);
        let counters = Arc::clone(&counters);
        std::thread::spawn(move || drain_usb(&handle, &mailbox, &counters))
    };
    let st = StreamState {
        exposure: Arc::clone(&lo.exposure),
        ae_on: Arc::clone(&lo.ae_on),
        led_mask: Arc::clone(&lo.led_mask),
        applied_leds,
        counter: 0,
        bright_ema: 0.0,
        last_set: 0,
        target_mean: 70.0,
        dark_dropped: 0,
        counters,
        diag: DiagPrev::default(),
    };
    lo.stream = Some(Streaming {
        mailbox,
        drain: Some(drain),
        st,
    });
    0
}

/// Block for the next non-dark frame and copy up to `len` bytes of the
/// 1280x240 interleaved L|R image into `buf`. Returns the number of bytes
/// written (= min(FRAME_BYTES, len)), or -1 if not streaming / the stream
/// stopped. Applies any pending control changes as a side effect.
///
/// # Safety
/// `h` must come from `leap_open_open`; `buf` must point to >= `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn leap_open_next_frame(h: *mut LeapOpen, buf: *mut u8, len: usize) -> i32 {
    let lo = match h.as_mut() {
        Some(l) => l,
        None => return -1,
    };
    let handle = Arc::clone(&lo.handle);
    let s = match lo.stream.as_mut() {
        Some(s) => s,
        None => return -1,
    };
    loop {
        match s.mailbox.take_latest() {
            None => return -1, // stopped
            Some(frame) => {
                if s.st.process(&handle, &frame).is_some() {
                    if buf.is_null() {
                        return -1;
                    }
                    let n = frame.len().min(len);
                    std::ptr::copy_nonoverlapping(frame.as_ptr(), buf, n);
                    return n as i32;
                }
                // dark frame: loop for the next one
            }
        }
    }
}

/// Set a fixed exposure (microseconds); also turns auto-exposure OFF.
/// # Safety
/// `h` must come from `leap_open_open`.
#[no_mangle]
pub unsafe extern "C" fn leap_open_set_exposure(h: *mut LeapOpen, us: u32) {
    if let Some(lo) = h.as_ref() {
        lo.exposure.store(us.clamp(10, 8000), Ordering::Relaxed);
        lo.ae_on.store(false, Ordering::Relaxed);
    }
}

/// Enable/disable closed-loop auto-exposure.
/// # Safety
/// `h` must come from `leap_open_open`.
#[no_mangle]
pub unsafe extern "C" fn leap_open_set_ae(h: *mut LeapOpen, on: bool) {
    if let Some(lo) = h.as_ref() {
        lo.ae_on.store(on, Ordering::Relaxed);
    }
}

/// Set the IR LED mask (bit0=left, bit1=center, bit2=right).
/// # Safety
/// `h` must come from `leap_open_open`.
#[no_mangle]
pub unsafe extern "C" fn leap_open_set_leds(h: *mut LeapOpen, mask: u8) {
    if let Some(lo) = h.as_ref() {
        lo.led_mask
            .store(u32::from(mask & 0b111), Ordering::Relaxed);
    }
}

/// Stop streaming, release the device back to the kernel, and free the handle.
/// # Safety
/// `h` must come from `leap_open_open` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn leap_open_close(h: *mut LeapOpen) {
    if h.is_null() {
        return;
    }
    let lo = Box::from_raw(h);
    let LeapOpen { handle, stream, .. } = *lo;
    if let Some(mut s) = stream {
        s.mailbox.stop();
        if let Some(d) = s.drain.take() {
            let _ = d.join();
        }
    }
    // the drain thread's Arc clone is gone after join, so this is the sole
    // owner and we can hand the interfaces back to the kernel explicitly.
    if let Ok(handle) = Arc::try_unwrap(handle) {
        for iface in [VC_IFACE, VS_IFACE] {
            let _ = handle.release_interface(iface);
            let _ = handle.attach_kernel_driver(iface);
        }
    }
}
