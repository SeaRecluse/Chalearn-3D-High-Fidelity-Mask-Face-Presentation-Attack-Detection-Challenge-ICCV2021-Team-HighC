import os
import glob
import shutil
import random

# from_path = "./avg/"
# to_path = "./sort_avg/"
from_path = "./train_upper/"
to_path = "./sort/"
do_shuffle = True
# do_shuffle = False
cut_scale = 0.8

if os.path.exists(to_path):
	shutil.rmtree(to_path)

os.makedirs(to_path + "train/")
os.makedirs(to_path + "test/")

if os.path.exists(from_path):
	file_list = os.listdir(from_path)
	for per_file in file_list:
		os.makedirs(to_path + "train/" + per_file)
		os.makedirs(to_path + "test/" + per_file)
		img_list = os.listdir(from_path + per_file)

		# img_list.sort()
		img_list.sort(reverse=True)

		if do_shuffle:
			random.shuffle(img_list)

		cut_len = (int)(len(img_list) * cut_scale)
		train_list = img_list[ : cut_len]
		test_list = img_list[cut_len + 1 : ]

		for img_path in train_list:
			shutil.copyfile(from_path + per_file + "/" + img_path,
						to_path + "train/" + per_file + "/" + img_path)
			print(to_path + "train/" + per_file + "/" + img_path)
		for img_path in test_list:
			shutil.copyfile(from_path + per_file + "/" + img_path,
						to_path + "test/" + per_file + "/" + img_path)
			print(to_path + "train/" + per_file + "/" + img_path)





