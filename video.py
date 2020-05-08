import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

from PIL import Image

def scale_0_1(img):
    max_v = np.max(img)
    min_v = np.min(img)
    m_val = max(np.absolute(max_v), np.absolute(min_v))
    return np.absolute(img)/m_val

def cmap_image(perturb):
    # print(perturb.shape)
    scaled = scale_0_1(perturb)
    # print(scaled)
    im = cm.jet(scaled, bytes=True)
    a = cm.jet(0, bytes=True)
    for i in range(perturb.shape[0]):
        for k in range(perturb.shape[1]):
            if np.array_equal(im[i, k], a):
                im[i, k, 3] = 0
            else:
                im[i, k, 3] = 150
    return Image.fromarray(im)

def viz_perturb(orig, perturbed):
    background = Image.fromarray(cm.gray(orig, bytes=True))
    foreground = cmap_image(perturbed)
    new_img = background.copy()
    background.paste(foreground, (0, 0), foreground)
    # background.show()
    return background, new_img

def write_video(orig_frames, perturb_frames, title="test"):
    videodims = (84, 84)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    FPS = 8
    n_frames = len(orig_frames)
    video = cv2.VideoWriter("{}.mp4".format(title), fourcc, FPS, videodims)
    for n in range(0, n_frames, 4):
        imtemp = viz_perturb(orig_frames[n], perturb_frames[n])[0].copy()
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGBA2RGB))
    video.release()



