#! /bin/bash

ffmpeg -y \
    -r 8 \
    -i coarse_frames/gamma/%4d.png \
    -vf "scale=960:540,drawtext=text='Low Frame Rate (Gamma)':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10,fps=64" \
    gamma_lfr.mp4

ffmpeg -y \
    -r 64 \
    -i fine_frames/gamma/%4d.png \
    -vf "scale=960:540,drawtext=text='High Frame Rate (Gamma)':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10" \
    gamma_hfr.mp4

ffmpeg -y \
    -r 64 \
    -i fine_frames/linear/%4d.png \
    -vf "scale=960:540,drawtext=text='High Frame Rate (Linear)':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10" \
    linear_hfr.mp4

ffmpeg -y \
    -r 64 \
    -i ev_frames/%4d.png \
    -vf "scale=960:540,drawtext=text='Event Frames':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10" \
    ev.mp4

ffmpeg -y \
    -r 64 \
    -i gamma_lfr.mp4 \
    -i gamma_hfr.mp4 \
    -i linear_hfr.mp4 \
    -i ev.mp4 \
    -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
    -map "[v]" stacked.mp4
