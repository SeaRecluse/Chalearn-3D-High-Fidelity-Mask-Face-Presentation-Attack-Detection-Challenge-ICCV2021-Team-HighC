from __future__ import print_function
import numpy as np
import cv2 as cv
import os
import sys
from skimage import transform as trans

FACE_ALIGN_SIZE = 224
FACE_ALIGN_SCALE = FACE_ALIGN_SIZE / 112
OFFSET = 8

REFERENCE_FACIAL_POINTS = np.array([
    [ (30.2946 + OFFSET) * FACE_ALIGN_SCALE, 51.6963 * FACE_ALIGN_SCALE ],
    [ (65.5318 + OFFSET) * FACE_ALIGN_SCALE, 51.5014 * FACE_ALIGN_SCALE ],
    [ (48.0252 + OFFSET) * FACE_ALIGN_SCALE, 71.7366 * FACE_ALIGN_SCALE ],
    [ (33.5493 + OFFSET) * FACE_ALIGN_SCALE, 92.3655 * FACE_ALIGN_SCALE ],
    [ (62.7299 + OFFSET) * FACE_ALIGN_SCALE, 92.2041 * FACE_ALIGN_SCALE ]
], np.float32)

def similarTransform(landmarks_pred, shape_targets):
    B = shape_targets
    A = np.hstack((np.array(landmarks_pred), np.ones((len(landmarks_pred), 1))))
                
    a = np.row_stack((np.array([-A[0][1], -A[0][0], 0, -1]), np.array([
                     A[0][0], -A[0][1], 1, 0])))
    b=np.row_stack((-B[0][1],B[0][0]))

    for i in range(A.shape[0]-1):
        i += 1
        a = np.row_stack((a, np.array([-A[i][1], -A[i][0], 0, -1])))
        a = np.row_stack((a, np.array([A[i][0], -A[i][1], 1, 0])))
        b = np.row_stack((b,np.array([[-B[i][1]], [B[i][0]]])))
         
    X, res, rank, s = np.linalg.lstsq(a, b)
    cos = (X[0][0]).real.astype(np.float32)
    sin = (X[1][0]).real.astype(np.float32)
    t_x = (X[2][0]).real.astype(np.float32)
    t_y = (X[3][0]).real.astype(np.float32)
    scale = np.sqrt(np.square(cos)+np.square(sin))
    
    H = np.array([[cos, -sin, t_x], [sin, cos, t_y]])
    s = np.linalg.eigvals(H[:, :-1])
    R = s.max() / s.min()
    
    return H
   
def faceAlign(img, landmark):
    KEY_POINT = np.array([landmark], np.float32)
    KEY_POINT = np.resize(KEY_POINT, [5, 2])

    tform = trans.SimilarityTransform()
    tform.estimate(KEY_POINT, REFERENCE_FACIAL_POINTS)
    similar_trans_matrix = tform.params[0:2, :]
    
    # similar_trans_matrix = similarTransform(KEY_POINT, REFERENCE_FACIAL_POINTS)
    aligned_face = cv.warpAffine(img.copy(), similar_trans_matrix, (FACE_ALIGN_SIZE , FACE_ALIGN_SIZE))

    return aligned_face