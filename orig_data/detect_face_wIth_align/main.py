import cv2
import math
import numpy as np
from retinaface import RetinaFace
from facealign import faceAlign

thresh = 0.4
gpuid = -1 #if gpu, set 0
resize_size = 640
save_size = 224
use_face_align = 1
prehead = "../"

detector = RetinaFace('./mnet.25/mnet.25', 0, gpuid, 'net3')

def resizeImg(img):
    m_resizeX, m_resizeY = 1.0, 1.0;
    H, W, C = img.shape

    m_resizeX = m_resizeY = max(H / resize_size, W / resize_size) 

    img = cv2.resize(img, ((int)(W / m_resizeX), (int)(H / m_resizeY)))
    img = cv2.copyMakeBorder(img, 0, max(resize_size - H, 0), 0, max(resize_size - W, 0), 0)

    return img

def getMaxFace(img):
    detect_tab = 0
    box, landmark = cropFace(img)
    if len(box):
        detect_tab = 1

        if use_face_align:
            img = faceAlign(img, landmark)

        else:
            w = (box[3] - box[1]) + 1
            h = (box[2] - box[0]) + 1
            l = max(w, h)

            x0 = int(max(0, (int)(box[1] + box[3] - l) / 2))
            y0 = int(max(0, (int)(box[0] + box[2] - l) / 2))
            h = int(min(img.shape[0], l))
            w = int(min(img.shape[1], l))

            img = img[x0 : x0 + w, y0 : y0 + h]

    return img, detect_tab

def cropFace(img):
    im_shape = img.shape
    scales = [min(im_shape[0],im_shape[1]), max(im_shape[0],im_shape[1])]

    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

    if len(faces):
        idx = 0
        for i in range(1,len(faces)):
            boxI = faces[i].astype(np.int)
            areaI = (boxI[3] - boxI[1]) * (boxI[2] - boxI[0])
            boxIdx = faces[idx].astype(np.int)
            areaIdx = (boxIdx[3] - boxIdx[1]) * (boxIdx[2] - boxIdx[0])
            if areaI > areaIdx:
               idx = i

        box = faces[idx].astype(np.int)
        landmark = landmarks[idx].astype(np.int)

        return box, landmark
    else:
        return [], []




def detectFace(load_path, save_path, 
    save_noface_A = "../no_faces_A",
    txt_path = "../faceList.txt"):
    line_list = []
    with open(txt_path, "r") as f:
        line_list = f.readlines()
        for n in range(len(line_list)):
            line = "/" + line_list[n].replace("\n", "")
            if line:
                img = cv2.imread(prehead + load_path + line)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, detect_tab = getMaxFace(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(prehead + save_path  + line, img)
                if not detect_tab:
                    cv2.imwrite(save_noface_A + line.replace("/", "#"), img)

            print("Process: "+ str(n + 1) + "/" + str(len(line_list)) + " " + line)


def detectFaceWithLabel(load_path, save_path, label_path, tag,
    save_noface_T = "../no_faces_T", 
    save_noface_F = "../no_faces_F", 
    txt_path = "../faceList.txt"):
    
    label_dict = {}
    line_list = []
    with open(prehead + label_path, "r") as f:
        line_list = f.readlines()
        for n in range(len(line_list)):
            line = line_list[n].replace("\n", "")
            if line:
                line = line.split(" ")
                key = line[0].replace(tag, "")
                value = line[1]
                label_dict.update({key : value})


    with open(txt_path, "r") as f:
        line_list = f.readlines()
        for n in range(len(line_list)):
            line = "/" + line_list[n].replace("\n", "")
            if line:
                img = cv2.imread(prehead + load_path + "/" + line)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resizeImg(img)
                img, detect_tab = getMaxFace(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if detect_tab:
                    cv2.imwrite(prehead + save_path + "/" + line, img)
                else:
                    if label_dict[line] == "1":
                        cv2.imwrite(save_noface_T + "/" + line.replace("/", "#"), img)
                    else:
                        cv2.imwrite(save_noface_F + "/" + line.replace("/", "#"), img)
            print("Process: "+ str(n + 1) + "/" + str(len(line_list)) + " " + line)


# detectFaceWithLabel("train", "save_merges_train", "train_label.txt", tag = "train")
detectFace("val", "save_merges_val")
# detectFaceWithLabel("val", "save_merges_val", "val_label.txt", tag = "val")
# detectFace("test", "save_merges_test")