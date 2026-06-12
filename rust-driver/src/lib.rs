//! leap-open: drive the original Leap Motion Controller over libusb directly,
//! the way leapd does (bulk image transport + UVC extension-unit control),
//! bypassing the closed daemon.
//!
//! v0.1 milestone — `info`: open the device, dump its full descriptor tree,
//! locate the bulk IN (image) endpoint and the VideoControl interface that
//! owns the vendor extension units, then prove we can take the device away
//! from the kernel `uvcvideo` driver (detach + claim + release).
//!
//! Run with the Ultraleap service STOPPED (it owns the device otherwise):
//!   pkexec systemctl stop ultraleap-hand-tracking-service
//!   sudo ./leap-open info        # claim needs root for detach
//!
//! The bring-up sequence (bringup_seq.txt) was decoded from a usbmon capture
//! of leapd -- see reverse-engineering/FINDINGS.md for the full trail.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use rusb::{Context, Device, DeviceHandle, Direction, TransferType, UsbContext};

const LEAP_VID: u16 = 0xf182;
const LEAP_PID: u16 = 0x0003;

// Discovered by `info`: VideoControl=iface 0, VideoStreaming=iface 1,
// bulk IN image endpoint = 0x83.
const VC_IFACE: u8 = 0;
const VS_IFACE: u8 = 1;
const BULK_IN_EP: u8 = 0x83;

// Image geometry decoded from the bulk stream (row-autocorrelation +0.98):
// 640x240 per eye, transmitted interleaved L/R -> 1280x240, 8-bit gray.
const EYE_W: usize = 640;
const FRAME_H: usize = 240;
const INTER_W: usize = EYE_W * 2; // 1280
const FRAME_BYTES: usize = INTER_W * FRAME_H; // 307_200

pub mod ffi;

/// CLI entry point for the `openleap` binary. The library also exposes a C ABI
/// in [`ffi`] (`leap_open_*`) for embedding — e.g. the Monado `leap_open` driver.
pub fn cli_main() {
    let cmd = std::env::args().nth(1).unwrap_or_else(|| "info".into());
    let r = match cmd.as_str() {
        "info" => run_info(),
        "bringup" => run_bringup(),
        "stream" => run_stream(),
        "calib" => run_calib(),
        other => {
            eprintln!("unknown command '{other}'. commands: info, bringup, stream, calib");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_info() -> rusb::Result<()> {
    let ctx = Context::new()?;
    let (device, desc) = open_leap(&ctx)?;
    println!(
        "found Leap {:04x}:{:04x}  bus {} addr {}  bcdDevice {:x}.{:02x}",
        desc.vendor_id(),
        desc.product_id(),
        device.bus_number(),
        device.address(),
        desc.device_version().major(),
        desc.device_version().minor(),
    );

    // ---- descriptor tree + endpoint discovery -------------------------------
    let mut bulk_in: Option<(u8, u8)> = None; // (interface, endpoint addr)
    let mut bulk_out: Option<(u8, u8)> = None;
    let mut vc_interface: Option<u8> = None;

    for cfg_idx in 0..desc.num_configurations() {
        let cfg = device.config_descriptor(cfg_idx)?;
        println!(
            "\nconfig #{} value {}  ifaces {}  max power {} mA",
            cfg_idx,
            cfg.number(),
            cfg.num_interfaces(),
            cfg.max_power() as u32 * 2
        );
        for iface in cfg.interfaces() {
            for alt in iface.descriptors() {
                let class = alt.class_code();
                let sub = alt.sub_class_code();
                // UVC: class 0x0e (Video). VideoControl subclass 0x01,
                // VideoStreaming subclass 0x02.
                let label = match (class, sub) {
                    (0x0e, 0x01) => " [UVC VideoControl - owns extension units]",
                    (0x0e, 0x02) => " [UVC VideoStreaming - image endpoint]",
                    _ => "",
                };
                println!(
                    "  iface {} alt {}  class {:02x} sub {:02x} proto {:02x}  \
                     endpoints {}{}",
                    alt.interface_number(),
                    alt.setting_number(),
                    class,
                    sub,
                    alt.protocol_code(),
                    alt.num_endpoints(),
                    label
                );
                if class == 0x0e && sub == 0x01 {
                    vc_interface = Some(alt.interface_number());
                }
                for ep in alt.endpoint_descriptors() {
                    let dir = ep.direction();
                    let tt = ep.transfer_type();
                    println!(
                        "      ep 0x{:02x}  {:<3} {:<11} maxpkt {}  interval {}",
                        ep.address(),
                        match dir {
                            Direction::In => "IN",
                            Direction::Out => "OUT",
                        },
                        format!("{tt:?}"),
                        ep.max_packet_size(),
                        ep.interval(),
                    );
                    if tt == TransferType::Bulk {
                        match dir {
                            Direction::In => {
                                bulk_in.get_or_insert((alt.interface_number(), ep.address()));
                            }
                            Direction::Out => {
                                bulk_out.get_or_insert((alt.interface_number(), ep.address()));
                            }
                        }
                    }
                }
            }
        }
    }

    println!("\n--- discovery summary ---");
    println!("VideoControl interface : {vc_interface:?}");
    println!("bulk IN  (images)      : {bulk_in:?}");
    println!("bulk OUT               : {bulk_out:?}");

    // ---- prove we can take the device from the kernel -----------------------
    println!("\n--- claim test (needs root: detaches uvcvideo) ---");
    let mut handle = device.open()?;
    let mut claimed = Vec::new();
    for iface in [vc_interface, bulk_in.map(|(i, _)| i)]
        .into_iter()
        .flatten()
    {
        match detach_and_claim(&mut handle, iface) {
            Ok(detached) => {
                println!(
                    "  iface {iface}: claimed{}",
                    if detached {
                        " (detached kernel driver)"
                    } else {
                        ""
                    }
                );
                claimed.push(iface);
            }
            Err(e) => println!("  iface {iface}: FAILED to claim: {e}"),
        }
    }
    for iface in &claimed {
        let _ = handle.release_interface(*iface);
        // hand the device back to the kernel so normal operation resumes
        let _ = handle.attach_kernel_driver(*iface);
    }
    println!(
        "\nclaim test {}.  device released back to kernel.",
        if claimed.is_empty() {
            "FAILED"
        } else {
            "PASSED"
        }
    );
    if !claimed.is_empty() {
        println!("next: `bringup` will replay the extension-unit init and read the bulk endpoint.");
    }
    Ok(())
}

/// One captured control transfer to replay.
struct Transfer {
    request_type: u8,
    request: u8,
    value: u16,
    index: u16,
    data: Vec<u8>,
}

fn parse_seq(path: &str) -> std::io::Result<Vec<Transfer>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let h = |s: &str| u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0);
        let data = if f[4] == "-" {
            Vec::new()
        } else {
            (0..f[4].len() / 2)
                .map(|i| u8::from_str_radix(&f[4][i * 2..i * 2 + 2], 16).unwrap_or(0))
                .collect()
        };
        out.push(Transfer {
            request_type: h(f[0]) as u8,
            request: h(f[1]) as u8,
            value: h(f[2]) as u16,
            index: h(f[3]) as u16,
            data,
        });
    }
    Ok(out)
}

fn run_bringup() -> rusb::Result<()> {
    let seq_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "bringup_seq.txt".into());
    let secs: u64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let seq = parse_seq(&seq_path).map_err(|e| {
        eprintln!("cannot read sequence file '{seq_path}': {e}");
        rusb::Error::Other
    })?;
    println!("loaded {} control transfers from {seq_path}", seq.len());

    let ctx = Context::new()?;
    let (device, desc) = open_leap(&ctx)?;
    println!(
        "opening Leap on bus {} addr {} (max power {} mA)",
        device.bus_number(),
        device.address(),
        device
            .config_descriptor(0)
            .map(|c| c.max_power() as u32 * 2)
            .unwrap_or(0)
    );
    let _ = desc;
    let mut handle = device.open()?;

    // take both interfaces from the kernel
    for iface in [VC_IFACE, VS_IFACE] {
        detach_and_claim(&mut handle, iface)?;
    }
    // streaming interface to alt 0 (bulk endpoint lives there)
    let _ = handle.set_alternate_setting(VS_IFACE, 0);

    // ---- replay leapd's bring-up control sequence ---------------------------
    let timeout = Duration::from_millis(500);
    let mut ok = 0usize;
    let mut fail = 0usize;
    for (i, t) in seq.iter().enumerate() {
        let r = handle.write_control(
            t.request_type,
            t.request,
            t.value,
            t.index,
            &t.data,
            timeout,
        );
        match r {
            Ok(_) => ok += 1,
            Err(e) => {
                fail += 1;
                if fail <= 5 {
                    println!(
                        "  transfer #{i} (req 0x{:02x} val 0x{:04x} idx 0x{:04x}) failed: {e}",
                        t.request, t.value, t.index
                    );
                }
            }
        }
    }
    println!("bring-up replay: {ok} ok, {fail} failed");
    println!("(IR LEDs should be on now if bring-up worked)");

    // ---- stream: frame-synced bulk read + closed-loop auto-exposure ---------
    println!("\nstreaming bulk 0x{BULK_IN_EP:02x} for {secs}s (frame-sync + auto-exposure)...");

    let mut exposure_us: u32 = 400; // mid-range start
    let target_mean: f64 = 70.0;
    set_exposure(&handle, exposure_us);

    let read_timeout = Duration::from_millis(500);
    let deadline = Instant::now() + Duration::from_secs(secs);
    // Each bulk read returns one UVC payload: [hlen-byte header][pixels].
    // header[0]=bHeaderLength, header[1]=bmHeaderInfo (bit0=FrameID,
    // bit1=EndOfFrame, bit7=EndOfHeader). Frames are delimited by the EOF
    // bit; we strip the header from each payload and concatenate the pixels.
    let mut buf = vec![0u8; 64 * 1024];
    let mut cur: Vec<u8> = Vec::with_capacity(FRAME_BYTES + 16384);
    let mut frames: Vec<Vec<u8>> = Vec::new(); // each exactly FRAME_BYTES
    let mut raw_sample: Vec<u8> = Vec::new();
    let mut ae_log: Vec<(usize, u64, u32)> = Vec::new();
    let mut reads = 0u64;
    let mut payloads = 0u64;
    let mut eofs = 0u64;
    let mut header_len = 0usize;

    while Instant::now() < deadline {
        let n = match handle.read_bulk(BULK_IN_EP, &mut buf, read_timeout) {
            Ok(n) => n,
            Err(rusb::Error::Timeout) => 0,
            Err(e) => {
                println!("  bulk read error: {e}");
                break;
            }
        };
        if n == 0 {
            continue;
        }
        reads += 1;
        if raw_sample.len() < FRAME_BYTES + 64 {
            raw_sample.extend_from_slice(&buf[..n]);
        }
        if n < 2 {
            continue;
        }
        let hlen = buf[0] as usize;
        let info = buf[1];
        // recognize a UVC payload header: end-of-header bit set, sane length
        if (info & 0x80) == 0 || hlen < 2 || hlen > n {
            continue;
        }
        payloads += 1;
        header_len = hlen;
        cur.extend_from_slice(&buf[hlen..n]);
        // guard against a missed EOF (dropped payload): don't grow unbounded
        if cur.len() > FRAME_BYTES + 16384 {
            cur.clear();
            continue;
        }
        if info & 0x02 != 0 {
            // end of frame
            eofs += 1;
            if cur.len() >= FRAME_BYTES {
                let frame: Vec<u8> = cur[..FRAME_BYTES].to_vec();
                if frames.len().is_multiple_of(15) {
                    let mean = frame_mean(&frame);
                    ae_log.push((frames.len(), mean, exposure_us));
                    exposure_us = adjust_exposure(exposure_us, mean, target_mean);
                    set_exposure(&handle, exposure_us);
                }
                frames.push(frame);
            }
            cur.clear();
        }
    }

    println!(
        "captured {} frames | {reads} reads, {payloads} payloads (hdr {header_len}B), {eofs} EOFs",
        frames.len()
    );
    if !ae_log.is_empty() {
        println!("  auto-exposure trace (frame#, mean, exposure_us):");
        for (f, m, e) in ae_log.iter().take(12) {
            println!("    f{f:<4} mean {m:<3} exp {e}us");
        }
        println!("  final exposure: {exposure_us}us");
    }
    let _ = std::fs::write("bulk_sample.raw", &raw_sample);

    // release back to kernel
    for iface in [VC_IFACE, VS_IFACE] {
        let _ = handle.release_interface(iface);
        let _ = handle.attach_kernel_driver(iface);
    }

    if frames.len() >= 4 {
        save_frames(&frames);
    } else {
        println!(
            "\nonly {} frames — bring-up or sync may be off; bulk_sample.raw saved.",
            frames.len()
        );
    }
    Ok(())
}

/// Read the on-device camera calibration via the LMC's "property knocking"
/// backdoor (per leapuvc): write a memory address to the Processing Unit's
/// sharpness control, read the byte back from the saturation control, for
/// addresses 100..=255 -> 156 raw calibration bytes written to stdout.
/// Also dumps the VideoControl unit descriptors to stderr so we can confirm
/// the Processing Unit id and selectors.
fn run_calib() -> rusb::Result<()> {
    use std::io::Write as _;
    let seq_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "bringup_seq.txt".into());
    let seq = parse_seq(&seq_path).map_err(|_| rusb::Error::Other)?;

    let ctx = Context::new()?;
    let (device, _desc) = open_leap(&ctx)?;

    // ---- dump VideoControl class-specific (unit/terminal) descriptors -------
    // CS_INTERFACE = 0x24; subtypes: 2=INPUT_TERMINAL 5=PROCESSING_UNIT
    // 6=EXTENSION_UNIT. bUnitID is at byte 3 for PU/EU.
    if let Ok(cfg) = device.config_descriptor(0) {
        for iface in cfg.interfaces() {
            for alt in iface.descriptors() {
                if alt.class_code() == 0x0e && alt.sub_class_code() == 0x01 {
                    let extra = alt.extra();
                    let mut i = 0;
                    while i + 2 < extra.len() {
                        let len = extra[i] as usize;
                        if len == 0 || i + len > extra.len() {
                            break;
                        }
                        if extra[i + 1] == 0x24 {
                            let subtype = extra[i + 2];
                            let kind = match subtype {
                                1 => "HEADER",
                                2 => "INPUT_TERMINAL",
                                3 => "OUTPUT_TERMINAL",
                                5 => "PROCESSING_UNIT",
                                6 => "EXTENSION_UNIT",
                                _ => "other",
                            };
                            let unit_id = if len > 3 { extra[i + 3] } else { 0 };
                            eprintln!(
                                "  VC descr subtype {subtype} ({kind}) unitID {unit_id} len {len}"
                            );
                        }
                        i += len;
                    }
                }
            }
        }
    }

    let mut handle = device.open()?;
    for iface in [VC_IFACE, VS_IFACE] {
        detach_and_claim(&mut handle, iface)?;
    }
    let timeout = Duration::from_millis(500);
    let mut replayed = 0usize;
    for t in &seq {
        if handle
            .write_control(
                t.request_type,
                t.request,
                t.value,
                t.index,
                &t.data,
                timeout,
            )
            .is_ok()
        {
            replayed += 1;
        }
    }
    eprintln!(
        "calib: bring-up {replayed}/{} transfers; reading calibration...",
        seq.len()
    );

    let calib = read_calibration(&handle);

    // quick sanity: bytes 7..11 and 7+11*4.. are little-endian f32 focal
    // lengths; print so we can eyeball plausibility (~150-700 px).
    let f32_at = |o: usize| -> f32 {
        if o + 4 <= calib.len() {
            f32::from_le_bytes([calib[o], calib[o + 1], calib[o + 2], calib[o + 3]])
        } else {
            0.0
        }
    };
    eprintln!(
        "calib: 156 bytes read. sig=[{} {}] ver={} left_focal~{:.1} right_focal~{:.1}",
        calib.first().copied().unwrap_or(0),
        calib.get(1).copied().unwrap_or(0),
        calib.get(2).copied().unwrap_or(0),
        f32_at(28),          // left focalLength (byteOffset 7 floats -> 7*4=28)
        f32_at(28 + 17 * 4), // right focalLength (+17 floats)
    );

    for iface in [VC_IFACE, VS_IFACE] {
        let _ = handle.release_interface(iface);
        let _ = handle.attach_kernel_driver(iface);
    }

    // write raw bytes to stdout for the Python parser
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    out.write_all(&calib).map_err(|_| rusb::Error::Other)?;
    Ok(())
}

/// Continuous capture: bring the device up, then write each synced frame to
/// stdout (4-byte magic "LFRM" + 4-byte LE frame counter + FRAME_BYTES raw
/// interleaved pixels), forever, with auto-exposure. Stops when stdout closes
/// (the viewer exited). All logging goes to stderr so stdout stays binary.
fn run_stream() -> rusb::Result<()> {
    let seq_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "bringup_seq.txt".into());
    let seq = parse_seq(&seq_path).map_err(|e| {
        eprintln!("cannot read sequence file '{seq_path}': {e}");
        rusb::Error::Other
    })?;

    let handle = open_claimed()?;
    let replayed = replay_seq(&handle, &seq);
    eprintln!(
        "stream: bring-up replayed {replayed}/{} transfers; streaming...",
        seq.len()
    );

    // State shared with a stdin command thread. Commands (one per line):
    //   "exp <us>"   set a fixed exposure (also turns auto-exposure off)
    //   "ae <0|1>"   enable/disable auto-exposure
    //   "leds <0-7>" IR LED mask (bit0=left bit1=center bit2=right); fewer = less
    //                bus draw. AE compensates brightness, so low power is fine.
    let exposure = Arc::new(AtomicU32::new(200)); // user-tuned fixed default
    let ae_on = Arc::new(AtomicBool::new(false)); // manual exposure by default ("ae 1" to enable)
                                                  // left+right: even illumination for BOTH cameras (each eye sits next to
                                                  // one lit LED) without overdriving the device — user-tested default.
    let led_mask = Arc::new(AtomicU32::new(0b101));
    {
        let exposure = exposure.clone();
        let ae_on = ae_on.clone();
        let led_mask = led_mask.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let p: Vec<&str> = line.split_whitespace().collect();
                match p.as_slice() {
                    ["exp", v] => {
                        if let Ok(n) = v.parse::<u32>() {
                            exposure.store(n.clamp(10, 8000), Ordering::Relaxed);
                            ae_on.store(false, Ordering::Relaxed);
                        }
                    }
                    ["ae", v] => ae_on.store(*v == "1" || *v == "on", Ordering::Relaxed),
                    ["leds", v] => {
                        if let Ok(n) = v.parse::<u32>() {
                            led_mask.store(n & 0b111, Ordering::Relaxed);
                        }
                    }
                    _ => {}
                }
            }
        });
    }

    let applied_leds = led_mask.load(Ordering::Relaxed) as u8;
    set_leds(&handle, applied_leds);
    set_exposure(&handle, exposure.load(Ordering::Relaxed));
    eprintln!(
        "stream: LED mask {applied_leds:03b} (left+right default; 'leds N' to change, AE keeps brightness)"
    );

    // ---- decoupled drain/write: never block the USB endpoint on the pipe ----
    // The device pushes ~30 MB/s through a shallow internal FIFO. The old
    // single-threaded loop blocked on the viewer's ~64 KB pipe while writing
    // each 307 KB frame, so nobody was reading the bulk endpoint and frames
    // tore (diag showed short/128 > 128 with zero USB errors). Now a dedicated
    // thread does nothing but drain USB + reassemble into a 1-slot mailbox,
    // and this thread ships the latest frame to stdout — dropping frames when
    // the viewer is slow instead of ever stalling the drain.
    let handle = Arc::new(handle);
    let mailbox = Arc::new(Mailbox::default());
    let counters = Arc::new(Counters::default());
    let drain = {
        let handle = Arc::clone(&handle);
        let mailbox = Arc::clone(&mailbox);
        let counters = Arc::clone(&counters);
        std::thread::spawn(move || drain_usb(&handle, &mailbox, &counters))
    };

    let mut out = std::io::BufWriter::new(std::io::stdout());
    let mut st = StreamState {
        exposure,
        ae_on,
        led_mask,
        applied_leds,
        counter: 0,
        bright_ema: 0.0,
        last_set: 0,
        target_mean: 70.0,
        dark_dropped: 0,
        counters: Arc::clone(&counters),
        diag: DiagPrev::default(),
    };

    while let Some(frame) = mailbox.take_latest() {
        // process() applies controls/AE and tells us whether to forward this
        // frame; emitting it (here, to stdout) is the caller's job.
        if let Some(id) = st.process(&handle, &frame) {
            if out.write_all(b"LFRM").is_err()
                || out.write_all(&id.to_le_bytes()).is_err()
                || out.write_all(&frame).is_err()
                || out.flush().is_err()
            {
                eprintln!("stream: stdout closed, stopping");
                break; // viewer exited
            }
        }
    }
    mailbox.stop();
    let _ = drain.join();

    // the drain thread's handle clone is gone after join, so unwrap succeeds
    // and we can hand the interfaces back to the kernel explicitly.
    if let Ok(handle) = Arc::try_unwrap(handle) {
        for iface in [VC_IFACE, VS_IFACE] {
            let _ = handle.release_interface(iface);
            let _ = handle.attach_kernel_driver(iface);
        }
    }
    eprintln!(
        "stream: stopped after {} frames ({} dark, {} short, {} forced-boundary, {} slow-viewer drops, {} timeouts, {} errors)",
        st.counter,
        st.dark_dropped,
        counters.short.load(Ordering::Relaxed),
        counters.forced.load(Ordering::Relaxed),
        counters.overwritten.load(Ordering::Relaxed),
        counters.timeouts.load(Ordering::Relaxed),
        counters.errors.load(Ordering::Relaxed),
    );
    Ok(())
}

/// Frame-integrity counters shared between the drain and writer threads
/// (monotonic totals; the diag line prints per-128-frame deltas).
#[derive(Default)]
struct Counters {
    short: AtomicU64,       // frame closed with too few bytes (lost payloads)
    forced: AtomicU64,      // FID toggled without an EOF (lost EOF payload)
    overwritten: AtomicU64, // complete frames dropped because the viewer was slow
    timeouts: AtomicU64,    // bulk read timeouts (device stalled / no data)
    errors: AtomicU64,      // bulk read errors (bus/power problems)
}

/// One-slot frame handoff from the drain thread to the writer thread. The
/// writer only ever wants the newest frame, so a deposit overwrites an
/// unclaimed one (counted as a slow-viewer drop) rather than blocking.
#[derive(Default)]
struct Mailbox {
    inner: Mutex<MailboxInner>,
    cv: Condvar,
}

#[derive(Default)]
struct MailboxInner {
    frame: Option<Vec<u8>>,
    stop: bool,
}

impl Mailbox {
    fn put(&self, frame: Vec<u8>, c: &Counters) {
        let mut g = self.inner.lock().unwrap();
        if g.frame.replace(frame).is_some() {
            c.overwritten.fetch_add(1, Ordering::Relaxed);
        }
        self.cv.notify_all();
    }

    /// Block until a frame is available; None once the stream has stopped.
    fn take_latest(&self) -> Option<Vec<u8>> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(f) = g.frame.take() {
                return Some(f);
            }
            if g.stop {
                return None;
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    fn stop(&self) {
        self.inner.lock().unwrap().stop = true;
        self.cv.notify_all();
    }

    fn stopped(&self) -> bool {
        self.inner.lock().unwrap().stop
    }
}

/// USB drain thread: read bulk payloads and reassemble frames (EOF-delimited,
/// with FID-toggle recovery when an EOF payload is lost), deposit complete
/// frames in the mailbox. Does no other I/O, so the endpoint is serviced
/// continuously and the device FIFO never overflows on our account.
fn drain_usb<T: UsbContext>(handle: &DeviceHandle<T>, mb: &Mailbox, c: &Counters) {
    // Real-time priority: a payload arrives every ~470 us, and a normal-
    // priority thread gets descheduled for milliseconds when the viewer's
    // inference saturates the CPU — each such delay tears a frame. We run as
    // root (pkexec), so SCHED_FIFO is available; the drain then preempts any
    // normal load the instant data is ready. Priority 10 is modest — well
    // below kernel IRQ threads, above everything SCHED_OTHER.
    unsafe {
        let param = libc::sched_param { sched_priority: 10 };
        if libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param) == 0 {
            eprintln!("stream: drain thread at real-time priority (SCHED_FIFO 10)");
        } else {
            eprintln!("stream: real-time priority unavailable; continuing best-effort");
        }
    }

    let read_timeout = Duration::from_millis(500);
    let mut buf = vec![0u8; 64 * 1024];
    let mut cur: Vec<u8> = Vec::with_capacity(FRAME_BYTES + 16384);
    let mut cur_fid: Option<u8> = None; // frame-ID bit of the in-progress frame
    let mut consec_errors: u32 = 0;

    while !mb.stopped() {
        let n = match handle.read_bulk(BULK_IN_EP, &mut buf, read_timeout) {
            Ok(n) => {
                consec_errors = 0;
                n
            }
            Err(rusb::Error::Timeout) => {
                c.timeouts.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Err(e) => {
                c.errors.fetch_add(1, Ordering::Relaxed);
                consec_errors += 1;
                // try to clear a halted endpoint and keep going; only give up
                // if it's persistent (device truly gone / unplugged).
                let _ = handle.clear_halt(BULK_IN_EP);
                if consec_errors >= 16 {
                    eprintln!("stream: bulk read error (persistent): {e}");
                    break;
                }
                continue;
            }
        };
        if n < 2 {
            continue;
        }
        let hlen = buf[0] as usize;
        let info = buf[1];
        if (info & 0x80) == 0 || hlen < 2 || hlen > n {
            continue;
        }
        let fid = info & 0x01;
        let eof = info & 0x02 != 0;

        // Robust boundary: if the frame-ID toggled and we never saw an EOF,
        // the device dropped the previous frame's EOF payload. Close the
        // previous frame on its own merit instead of merging two into a tear.
        if let Some(prev) = cur_fid {
            if fid != prev && !cur.is_empty() {
                c.forced.fetch_add(1, Ordering::Relaxed);
                if cur.len() >= FRAME_BYTES {
                    mb.put(cur[..FRAME_BYTES].to_vec(), c);
                } else {
                    c.short.fetch_add(1, Ordering::Relaxed); // lost payload(s)
                }
                cur.clear();
            }
        }

        cur.extend_from_slice(&buf[hlen..n]);
        cur_fid = Some(fid);
        if cur.len() > FRAME_BYTES + 16384 {
            cur.clear();
            cur_fid = None;
            continue;
        }
        if eof {
            if cur.len() >= FRAME_BYTES {
                mb.put(cur[..FRAME_BYTES].to_vec(), c);
            } else if !cur.is_empty() {
                c.short.fetch_add(1, Ordering::Relaxed);
            }
            cur.clear();
            cur_fid = None;
        }
    }
    mb.stop();
}

/// Per-128-frame snapshots so the diag line reports *deltas*, not totals.
#[derive(Default)]
struct DiagPrev {
    dark: u64,
    short: u64,
    forced: u64,
    overwritten: u64,
    timeouts: u64,
    errors: u64,
}

/// Writer-thread state: exposure/LED control, brightness tracking, and frame
/// emission to stdout. Bundled into one struct so `finalize_frame` takes a
/// sane number of arguments (no clippy suppression).
struct StreamState {
    exposure: Arc<AtomicU32>,
    ae_on: Arc<AtomicBool>,
    led_mask: Arc<AtomicU32>,
    applied_leds: u8,
    counter: u32,
    bright_ema: f64, // running mean of non-dark frames
    last_set: u32,   // last exposure actually written to the device
    target_mean: f64,
    dark_dropped: u64,
    counters: Arc<Counters>,
    diag: DiagPrev,
}

impl StreamState {
    /// Process one complete frame: apply pending LED/exposure changes, run
    /// auto-exposure, drop strobe (dark) frames, and forward good frames to
    /// stdout as `LFRM`+counter+pixels. Returns false if stdout closed (stop).
    /// Apply pending LED/exposure/AE changes and classify the frame. Returns
    /// `Some(frame_id)` if it should be forwarded, `None` if it was a dark
    /// (LED-off strobe) frame and dropped. Emitting the frame — stdout for the
    /// CLI, a caller buffer for the C ABI — is the caller's responsibility.
    fn process<T: UsbContext>(&mut self, handle: &DeviceHandle<T>, frame: &[u8]) -> Option<u32> {
        // apply an LED-mask change requested over stdin / the C ABI
        let want_leds = self.led_mask.load(Ordering::Relaxed) as u8;
        if want_leds != self.applied_leds {
            set_leds(handle, want_leds);
            self.applied_leds = want_leds;
        }

        let mean = frame_mean(frame) as f64;
        // dark-frame (LED-off strobe) detection: much darker than recent
        let is_dark = self.bright_ema > 5.0 && mean < 0.4 * self.bright_ema;
        if !is_dark {
            self.bright_ema = if self.bright_ema == 0.0 {
                mean
            } else {
                0.9 * self.bright_ema + 0.1 * mean
            };
        }
        if self.ae_on.load(Ordering::Relaxed) && !is_dark && self.counter.is_multiple_of(10) {
            let e = self.exposure.load(Ordering::Relaxed);
            self.exposure.store(
                adjust_exposure(e, mean as u64, self.target_mean),
                Ordering::Relaxed,
            );
        }
        let want = self.exposure.load(Ordering::Relaxed);
        if want != self.last_set {
            set_exposure(handle, want);
            self.last_set = want;
        }

        if is_dark {
            self.dark_dropped += 1; // don't forward dark frames
            return None;
        }
        let id = self.counter;
        self.counter = self.counter.wrapping_add(1);

        // diagnostic every ~128 forwarded frames, reported as deltas so a burst
        // of tearing/timeouts stands out against an otherwise clean stream.
        if self.counter.is_multiple_of(128) {
            let c = &self.counters;
            let (short, forced, over, tos, errs) = (
                c.short.load(Ordering::Relaxed),
                c.forced.load(Ordering::Relaxed),
                c.overwritten.load(Ordering::Relaxed),
                c.timeouts.load(Ordering::Relaxed),
                c.errors.load(Ordering::Relaxed),
            );
            let d = &mut self.diag;
            eprintln!(
                "diag: mean~{:.0} exp={}us leds={:03b} dark/128={} short/128={} \
                 forced/128={} drop/128={} timeouts/128={} errors/128={} ae={}",
                self.bright_ema,
                want,
                self.applied_leds,
                self.dark_dropped - d.dark,
                short - d.short,
                forced - d.forced,
                over - d.overwritten,
                tos - d.timeouts,
                errs - d.errors,
                self.ae_on.load(Ordering::Relaxed) as u8
            );
            *d = DiagPrev {
                dark: self.dark_dropped,
                short,
                forced,
                overwritten: over,
                timeouts: tos,
                errors: errs,
            };
        }
        Some(id)
    }
}

/// Set which IR LEDs are on via a 3-bit mask (bit0=left, bit1=center,
/// bit2=right). The LMC overloads the Processing Unit contrast control (unit 5,
/// selector 0x03) as an LED switch: value = (sub_selector | (on << 6)), with
/// sub 2/3/4 = left/center/right. Each lit LED adds ~current+heat, so fewer
/// LEDs = lower bus draw. All three at once can brown out a weak USB port.
fn set_leds<T: UsbContext>(handle: &DeviceHandle<T>, mask: u8) {
    for (i, led) in [2u32, 3, 4].into_iter().enumerate() {
        let on = (mask >> i) & 1;
        let v = led | (u32::from(on) << 6);
        let _ = handle.write_control(
            0x21,
            0x01,
            0x0300,
            0x0500,
            &v.to_le_bytes(),
            Duration::from_millis(200),
        );
    }
}

/// Set sensor exposure (microseconds) via UVC SET_CUR on extension unit 2,
/// selector 0x0b (zoom-absolute, which the LMC maps to exposure).
fn set_exposure<T: UsbContext>(handle: &DeviceHandle<T>, us: u32) {
    let data = us.to_le_bytes();
    // wValue = selector<<8 = 0x0b00 ; wIndex = (unit<<8)|interface = 0x0200
    let _ = handle.write_control(
        0x21,
        0x01,
        0x0b00,
        0x0200,
        &data,
        Duration::from_millis(200),
    );
}

/// Multiplicative exposure control toward a target mean brightness.
/// Clamped to [10, 8000] us: the sensor keeps integrating up to ~one frame
/// period (~8700 us at ~115 fps), so higher exposures DO keep brightening
/// (especially with fewer LEDs lit / no display gain) — only past the frame
/// period does it cap and start dropping the frame rate.
fn adjust_exposure(cur: u32, mean: u64, target: f64) -> u32 {
    let mean = (mean.max(1)) as f64;
    let ratio = (target / mean).clamp(0.6, 1.8);
    ((cur as f64 * ratio) as u32).clamp(10, 8000)
}

fn frame_mean(frame: &[u8]) -> u64 {
    if frame.is_empty() {
        return 0;
    }
    frame.iter().map(|&b| b as u64).sum::<u64>() / frame.len() as u64
}

/// De-interleave and write frames as PGM (P5) — no image-crate dependency.
/// Prefers well-exposed frames (mean in [40,140]) so hand-present frames are
/// saved even when auto-exposure spent time on an empty/dark scene; samples up
/// to 16 of them spread evenly through the capture.
fn save_frames(frames: &[Vec<u8>]) {
    let n = frames.len();
    let outdir = Path::new("captured_frames");
    let _ = std::fs::create_dir_all(outdir);
    let well: Vec<usize> = (0..n)
        .filter(|&i| {
            let m = frame_mean(&frames[i]);
            (40..=140).contains(&m)
        })
        .collect();
    let pool: Vec<usize> = if well.is_empty() {
        (0..n).collect()
    } else {
        well
    };
    println!(
        "\nsaving frames to {}/ ({n} total, {} well-exposed)",
        outdir.display(),
        pool.len()
    );
    let want = 16.min(pool.len());
    let picks: Vec<usize> = (0..want)
        .map(|k| pool[(k * (pool.len().saturating_sub(1))) / want.max(1)])
        .collect();
    for (k, &fi) in picks.iter().enumerate() {
        let frame = &frames[fi];
        let mut left = vec![0u8; EYE_W * FRAME_H];
        let mut right = vec![0u8; EYE_W * FRAME_H];
        for row in 0..FRAME_H {
            let src = &frame[row * INTER_W..(row + 1) * INTER_W];
            for col in 0..EYE_W {
                left[row * EYE_W + col] = src[col * 2];
                right[row * EYE_W + col] = src[col * 2 + 1];
            }
        }
        write_pgm(
            &outdir.join(format!("frame{k}_idx{fi}_left.pgm")),
            &left,
            EYE_W,
            FRAME_H,
        );
        write_pgm(
            &outdir.join(format!("frame{k}_idx{fi}_right.pgm")),
            &right,
            EYE_W,
            FRAME_H,
        );
        let lmean = frame_mean(&left);
        println!("  frame {k} (idx {fi}): left mean {lmean}");
    }
    println!("view: imv/feh, or `for f in captured_frames/*.pgm; do convert $f $f.png; done`");
}

fn write_pgm(path: &Path, data: &[u8], w: usize, h: usize) {
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = write!(f, "P5\n{w} {h}\n255\n");
        let _ = f.write_all(data);
    }
}

fn detach_and_claim<T: UsbContext>(handle: &mut DeviceHandle<T>, iface: u8) -> rusb::Result<bool> {
    let mut detached = false;
    if handle.kernel_driver_active(iface).unwrap_or(false) {
        handle.detach_kernel_driver(iface)?;
        detached = true;
    }
    handle.claim_interface(iface)?;
    Ok(detached)
}

fn open_leap<T: UsbContext>(ctx: &T) -> rusb::Result<(Device<T>, rusb::DeviceDescriptor)> {
    for device in ctx.devices()?.iter() {
        let desc = device.device_descriptor()?;
        if desc.vendor_id() == LEAP_VID && desc.product_id() == LEAP_PID {
            return Ok((device, desc));
        }
    }
    Err(rusb::Error::NoDevice)
}

/// Open the Leap, take both interfaces from the kernel, select the streaming
/// alt setting. No bring-up replay yet. Shared by the CLI stream path and the
/// C ABI ([`ffi`]). The returned handle owns its libusb context.
fn open_claimed() -> rusb::Result<DeviceHandle<Context>> {
    let ctx = Context::new()?;
    let (device, _desc) = open_leap(&ctx)?;
    let mut handle = device.open()?;
    for iface in [VC_IFACE, VS_IFACE] {
        detach_and_claim(&mut handle, iface)?;
    }
    let _ = handle.set_alternate_setting(VS_IFACE, 0);
    Ok(handle)
}

/// Replay leapd's extension-unit bring-up control writes; returns how many the
/// device accepted.
fn replay_seq<T: UsbContext>(handle: &DeviceHandle<T>, seq: &[Transfer]) -> usize {
    let timeout = Duration::from_millis(500);
    let mut replayed = 0usize;
    for t in seq {
        if handle
            .write_control(
                t.request_type,
                t.request,
                t.value,
                t.index,
                &t.data,
                timeout,
            )
            .is_ok()
        {
            replayed += 1;
        }
    }
    replayed
}

/// Read the 156-byte on-device stereo calibration via the Processing Unit
/// "property knock": write an address to selector 0x08, read the byte back from
/// 0x07, for addresses 100..256. The device must already be brought up. Unit 5,
/// 4-byte controls.
fn read_calibration<T: UsbContext>(handle: &DeviceHandle<T>) -> Vec<u8> {
    let timeout = Duration::from_millis(500);
    let pu_index: u16 = (5 << 8) | u16::from(VC_IFACE); // 0x0500
    let sharp: u16 = 0x0800; // selector 0x08 << 8 (write address)
    let sat: u16 = 0x0700; // selector 0x07 << 8 (read byte)
    let mut calib = Vec::with_capacity(156);
    for addr in 100u32..256 {
        let _ = handle.write_control(0x21, 0x01, sharp, pu_index, &addr.to_le_bytes(), timeout);
        let mut buf = [0u8; 4];
        let n = handle
            .read_control(0xa1, 0x81, sat, pu_index, &mut buf, timeout)
            .unwrap_or(0);
        calib.push(if n > 0 { buf[0] } else { 0 });
    }
    calib
}
