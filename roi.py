import cv2
import numpy as np

from skimage import measure
from imutils import contours, is_cv2

def roi(image, threshold=200, padding=5, min_size=10, max_size=2000):
    """Returns a list of ROI tuples on the form ((x, y), (w, h))"""
    regions_of_interest = list()

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)[1]
    dilate = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)

    labels = measure.label(dilate, neighbors=4, background=0)
    mask = np.zeros(dilate.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(dilate.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if min_size < numPixels < max_size:
            mask = cv2.add(mask, labelMask)

    try: 
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
    except:
        return []

    height, width, _ = image.shape

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        x = int(max(0, x - padding))
        y = int(max(0, y - padding))
        w = int(min(w + padding*2, width - x - 1))
        h = int(min(h + padding*2, height - y - 1))
        regions_of_interest.append(((x, y), (w, h)))

    return regions_of_interest
