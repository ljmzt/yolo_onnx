from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms as T

def mydisplay(img, bbox_output, class_output, p_output):
    plt.figure(figsize=(10,10))
    img = np.array(T.ToPILImage()(img))
    plt.imshow(img)

    for bbox, class_name, p in zip(bbox_output, class_output, p_output):
        H, W = img.shape[:2]
        hx, hy, hw, hh = bbox
        xmin = (hx - 0.5 * hw) * W
        xmax = (hx + 0.5 * hw) * W
        ymin = (hy - 0.5 * hh) * H
        ymax = (hy + 0.5 * hh) * H
    
        plt.plot([xmin, xmax], [ymin, ymin], 'r-')
        plt.plot([xmax, xmax], [ymin, ymax], 'r-')
        plt.plot([xmax, xmin], [ymax, ymax], 'r-')
        plt.plot([xmin, xmin], [ymax, ymin], 'r-')

        text = "{0:s} {1:5.3f}".format(class_name, p)
        plt.text(xmin, ymin, text)