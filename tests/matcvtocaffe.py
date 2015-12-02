import numpy as np

def format_frame(frame):
    img = frame.astype(np.float32)/255.
    img = img[...,::-1]
    return img
