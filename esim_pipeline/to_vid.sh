ffmpeg -framerate 90 -pattern_type glob -i 'generated_data/enerf_carpet/intensity_imgs/*.png' -c:v libx264 -pix_fmt yuv420p out_1.mp4
