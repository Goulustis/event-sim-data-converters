import numpy as np
import cv2
import os.path as osp
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def evs_to_img(evs:np.ndarray, abs = True):
    evs = evs.squeeze()
    if len(evs.shape) == 2:
        evs = evs[None]

    frames = np.zeros((*evs.shape, 3), dtype=np.uint8)
    pos_loc = evs > 0
    neg_loc = evs < 0

    if abs:
        frames[pos_loc, 1] = 255
        frames[neg_loc, 0] = 255
    else:
        to_uint8 = lambda x : (x/x.max() * 255).astype(np.uint8)
        pos_evs, neg_evs = evs[pos_loc], np.abs(evs[neg_loc])
        pos_evs, neg_evs = to_uint8(pos_evs), to_uint8(neg_evs)
        frames[pos_loc, 1] = pos_evs
        frames[neg_loc, 0] = neg_evs

    return frames


def write_num(frame, number):
    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    # Determine the size of the text
    text_size, _ = cv2.getTextSize(str(number), font, font_scale, line_type)

    # Define the position of the text on the frame
    text_position = (10, text_size[1] + 10)  # Adjust position as needed

    # Write the number on the frame
    cv2.putText(frame, str(number), text_position, font, font_scale, font_color, line_type)

    return frame

def create_ev_vid(add_frame_num=True):
    eimg_f = "/home/hunter/projects/esim_pipeline/adapt_carpet_formatted/ecam_set/eimgs/eimgs_1x.npy"
    print("creating eimg vid")
    ev_vid = evs_to_img(np.load(eimg_f))
    if add_frame_num:
        for i, frame in enumerate(ev_vid):
            ev_vid[i] = write_num(frame, i)

    print("create done")
    h, w = ev_vid[0].shape[:2]
    # resize = lambda x : cv2.resize(x, (w//2, h//2), interpolation = cv2.INTER_AREA)
    # ev_vid = [resize(e) for e in ev_vid]
    ev_vid = [e for e in ev_vid]
    
    clip = ImageSequenceClip(ev_vid, fps=60)
    clip.write_videofile("adapt_carpet.mp4")

    for i, frame in enumerate(ev_vid):
        dst_f = osp.join("ev_frames", str(i).zfill(5) + ".png")
        cv2.imwrite(dst_f, frame)


if __name__ == "__main__":
    create_ev_vid()