import os
import shutil

# from_path = "train/"
# from_path = "save_merges_train/"

from_path = "val/"
save_path = "save_merges_val/"

imgs_txt = "faceList.txt"
rank_nums = 2
face_num_limit_min = 0
face_num_limit_max = 99999

def copy_folder(name_list):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    if os.path.exists("./no_faces_T"):
        shutil.rmtree("./no_faces_T")
    os.makedirs("./no_faces_T")

    if os.path.exists("./no_faces_F"):
        shutil.rmtree("./no_faces_F")
    os.makedirs("./no_faces_F")

    if os.path.exists(from_path):
        for per_name in name_list:
            if rank_nums == 2:
                img_list = os.listdir(from_path + per_name)
                if len(img_list) > face_num_limit_min \
                    and len(img_list) < face_num_limit_max:
                        os.makedirs(save_path + per_name)
            

if os.path.exists(imgs_txt):
    os.remove(imgs_txt)

name_list = os.listdir(from_path)
f = open(imgs_txt, "a", newline = "\n")

for per_name in name_list:
    if rank_nums == 1:
        f.write(per_name + "\n")
    elif rank_nums == 2:
        img_list = os.listdir(from_path + per_name)
        if len(img_list) > face_num_limit_min \
            and len(img_list) < face_num_limit_max:
            for per_img in img_list:
                f.write(per_name + "/" + per_img + "\n")
f.close()

copy_folder(name_list)

