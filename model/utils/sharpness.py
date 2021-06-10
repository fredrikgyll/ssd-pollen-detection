import cv2
import numpy as np
import numpy.fft as fft


def crop(img, bbox, padding=0):
    h, w, *_ = img.shape
    x1, y1, x2, y2 = bbox[:4]
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, w)
    y2 = min(y2 + padding, h)
    return img[y1:y2, x1:x2]


def bbox_crops(img, bboxes, padding=0):
    return [crop(img, bbox, padding) for bbox in bboxes]

def image_quality(img: np.ndarray):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img.shape
    fft_img = np.abs(fft.fftshift(fft.fft2(img)))
    threshhold = np.max(fft_img) / 1000
    t_h = np.sum(fft_img > threshhold)
    return t_h / (m * n)


def image_bbox_quality(img: np.ndarray, bboxes: np.ndarray):
    box_crops = bbox_crops(img, bboxes)
    qualities = [image_quality(im) for im in box_crops]
    return qualities

