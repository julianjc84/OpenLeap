#!/usr/bin/env bash
# Live skeleton viewer for the open Leap Motion stack.
#
#   1. Plug in the Leap Motion Controller.
#   2. Run this script:  ./run_live.sh
#   3. Authorize the single polkit prompt (it stops the closed Ultraleap
#      service and starts our open streamer).
#   4. A window opens with the live IR feed + skeleton overlay. Move your hand.
#      Controls: eye (L/R), rotation, flip H/V, gain, crop.
#
# The closed ultraleap-hand-tracking-service is left stopped on exit (start it
# again with: pkexec systemctl start ultraleap-hand-tracking-service).
cd "$(dirname "$0")"
# Let GTK use GPU (GL) rendering when run from your own session — much faster
# than software. If you ever see a blank window / GL errors, force software with:
#   GSK_RENDERER=cairo ./run_live.sh
exec .venv/bin/python tracking/live_viewer.py "$@"
