from typing import Literal

import cv2
import numpy as np
from PIL import Image

RELATIVE_DIST_EYES = (1,0)
RELATIVE_DIST_EYE_NOSE = (0.485, 0.805)
RELATIVE_PADDING_X= (0.7,0.7)
RELATIVE_PADDING_Y= (1.5,0.4)
IMAGE_SIZE = (224, 224)

def _rescale_coord(value:float, axis:Literal[0,1]):
    if axis == 0:
        size = RELATIVE_DIST_EYES[0] + sum(RELATIVE_PADDING_X)
    elif axis == 1:
        size = RELATIVE_DIST_EYE_NOSE[1] + sum(RELATIVE_PADDING_Y)
    else:
        raise Exception("axis should be 0 or 1")
    return value*IMAGE_SIZE[axis]/size


def get_ref_pts():
   left_eye_coord = np.array([_rescale_coord(RELATIVE_PADDING_X[0],0),
                              _rescale_coord(RELATIVE_PADDING_Y[0],1)])
   right_eye_coord = np.array([_rescale_coord(RELATIVE_PADDING_X[0] + RELATIVE_DIST_EYES[0],0),
                               _rescale_coord(RELATIVE_PADDING_Y[0], 1)])
   nose_coord = np.array([_rescale_coord(RELATIVE_PADDING_X[0] + RELATIVE_DIST_EYE_NOSE[0],0),
                          _rescale_coord(RELATIVE_PADDING_Y[0] + RELATIVE_DIST_EYE_NOSE[1], 1)])
   return left_eye_coord, right_eye_coord, nose_coord

def align_face(image:Image.Image, src_pts):
    image = np.array(image)
    dst_pts = np.array(get_ref_pts())

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    aligned = cv2.warpAffine(
        image,
        M,
        IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return Image.fromarray(aligned)