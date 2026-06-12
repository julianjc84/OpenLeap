#!/usr/bin/env bash
# Standalone Leap Open -> Monado/Mercury hand-tracking harness (no
# monado-service, no compositor, no OpenXR — just the tracking pipeline).
#
#   1. Plug in the Leap Motion Controller.
#   2. Run this script:  ./run_monado.sh
#   3. Authorize the single polkit prompt (it stops the closed Ultraleap
#      service and claims the device as root, which the USB bring-up needs).
#   4. Mercury's tracked 3D hand joints print as you move a hand into view:
#         [L] wrist=(x, y, z) m   pinch=NN mm
#      Ctrl-C to stop. The closed ultraleap service is left stopped (restart
#      with: pkexec systemctl start ultraleap-hand-tracking-service).
#
# Reads: rust-driver/bringup_seq.txt, tracking/calib_monado.json, and
# hand-tracking-models/ (Mercury's ONNX nets). The Monado fork is expected at
# ../monado (override with MONADO_DIR=...).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Mode: "headless" (default) prints joints; "gui" opens monado-gui's debug
# window — click "Hand-Tracking Demo" for the live IR preview + exposure/LED
# sliders + Mercury's hand overlay.
MODE="${1:-headless}"

MONADO_DIR="${MONADO_DIR:-$HERE/../monado}"
BIN="$MONADO_DIR/build/src/xrt/targets/leap_open_ht/leap_open_ht"
BIN_GUI="$MONADO_DIR/build/src/xrt/targets/gui/monado-gui"
SEQ="$HERE/rust-driver/bringup_seq.txt"
CALIB="$HERE/tracking/calib_monado.json"
MODELS="$HERE/hand-tracking-models"

# --- regenerate the Monado calibration if it's missing -----------------------
if [ ! -e "$CALIB" ] && [ -e "$HERE/tracking/calibration.bin" ]; then
	echo "generating $CALIB from the saved on-device calibration ..."
	( cd "$HERE/tracking" && "$HERE/.venv/bin/python" calibration.py monado calibration.bin -o calib_monado.json )
fi

# --- sanity ------------------------------------------------------------------
if [ ! -x "$BIN" ]; then
	echo "harness not built: $BIN" >&2
	echo "build it:  cmake -B build -DXRT_BUILD_DRIVER_LEAP_OPEN=ON ... ; ninja -C build leap_open_ht  (in $MONADO_DIR)" >&2
	exit 1
fi
for f in "$SEQ" "$CALIB" "$MODELS/grayscale_keypoint_jan18.onnx"; do
	[ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

# --- make Mercury find the ONNX models without a system install --------------
# ht_device_create searches $XDG_DATA_HOME/monado/hand-tracking-models first, so
# stage a repo-owned symlink (no root needed) and point XDG_DATA_HOME at it.
DATA="$HERE/.monado-data"
mkdir -p "$DATA/monado"
ln -sfn "$MODELS" "$DATA/monado/hand-tracking-models"

if [ "$MODE" = "gui" ]; then
	[ -x "$BIN_GUI" ] || { echo "monado-gui not built: $BIN_GUI (ninja -C build gui)" >&2; exit 1; }
	echo "Leap Open -> Mercury (GUI).  Authorize the prompt, then click \"Hand-Tracking Demo\"."
	echo "You'll get: Left/Right IR previews, exposure/AE/LED sliders, and the tracked hand."
	# The GUI runs as root (USB claim) but must reach your session's display, so
	# forward the display env (works for X11/XWayland via XAUTHORITY and for
	# Wayland via XDG_RUNTIME_DIR; root can open the user's socket).
	exec pkexec sh -c "
		systemctl stop ultraleap-hand-tracking-service 2>/dev/null
		exec env \
			DISPLAY='${DISPLAY:-}' \
			WAYLAND_DISPLAY='${WAYLAND_DISPLAY:-}' \
			XDG_RUNTIME_DIR='${XDG_RUNTIME_DIR:-}' \
			XAUTHORITY='${XAUTHORITY:-}' \
			XDG_DATA_HOME='$DATA' \
			LEAP_OPEN_SEQ='$SEQ' \
			LEAP_OPEN_CALIB='$CALIB' \
			'$BIN_GUI'
	"
fi

echo "Leap Open -> Mercury (headless).  bin=$BIN"
echo "Plug in the Leap, authorize the prompt, move a hand into view. Ctrl-C to stop."
echo "(For the visual UI instead: ./run_monado.sh gui)"

# pkexec strips the environment, so set the vars inside the elevated shell.
exec pkexec sh -c "
	systemctl stop ultraleap-hand-tracking-service 2>/dev/null
	exec env \
		XDG_DATA_HOME='$DATA' \
		LEAP_OPEN_SEQ='$SEQ' \
		LEAP_OPEN_CALIB='$CALIB' \
		'$BIN'
"
