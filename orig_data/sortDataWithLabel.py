import cv2
import os
import shutil

# from_path = "./save_merges_train"
# from_tab = "train"
# from_txt = "./train_label.txt"
to_path = "./sort_train/"

from_path = "./save_merges_val"
from_tab = "val"
from_txt = "./val_label.txt"
to_path = "./sort_val/"


save_path_T = to_path + "1/"
save_path_F = to_path + "0/"

def moveImg(orig_path, to_path, save_count, tab):
	save_name = to_path + str(tab) + "/" + orig_path.replace(from_path, "").replace("/", "#")
	shutil.copyfile(orig_path, save_name)
	print(save_name)


if os.path.exists(to_path):
	shutil.rmtree(to_path)
os.makedirs(to_path)
os.makedirs(save_path_T)
os.makedirs(save_path_F)

line_list = []
T_count = 0
F_count = 0
with open(from_txt, "r") as f:
	line_list = f.readlines()

for n in range(len(line_list)):
	line = line_list[n]
	if line:
		line = line.replace(from_tab, from_path)
		line = line.replace("\n", "").split(" ")
		img_path = line[0]
		tab = int(line[1])

		print(img_path)
		if os.path.exists(img_path):
			if tab == 1:
				T_count += 1
				moveImg(img_path, to_path, T_count, tab)
			elif tab == 0:
				F_count += 1
				moveImg(img_path, to_path, F_count, tab)

		print(str(n + 1) + "/" + str(len(line_list)) + "\t")
