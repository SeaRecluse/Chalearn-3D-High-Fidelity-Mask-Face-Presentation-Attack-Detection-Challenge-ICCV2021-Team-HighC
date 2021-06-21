import os
import glob
import shutil
import random
import cv2
import numpy as np

from_path = "./sort_train/"
to_path = "./train_upper/"

# from_path = "./sort_val/"
# to_path = "./val_upper/"

do_shuffle = True

def getAvgList(img_data_list, save_path, save_count = 0, merge_nums = 1):
	if do_shuffle:
		random.shuffle(img_data_list)

	H, W, C = img_data_list[0].shape

	for i in range(0, len(img_data_list)- merge_nums, merge_nums):
		img_avg = np.zeros((H, W, C), dtype = np.float32)
		for n in range(merge_nums):
			img_avg += (img_data_list[i + n] / merge_nums)
		
		save_count += 1
		save_name = "avg_"
		for n in range(6 - len(str(save_count))):
			save_name += "0"
		save_name += (str(save_count) + ".png")
		cv2.imwrite(save_path + "/" + save_name, img_avg)

		print("count: " + str(save_count) + " merge_nums: " + str(merge_nums) + " process: "
			+ str(i // merge_nums + 1) + "/" + str(len(img_data_list) // merge_nums))

	return save_count

def getContrastAndBrightList(img_data_list, save_path, save_count = 0):
	if do_shuffle:
		random.shuffle(img_data_list)

	for i in range(len(img_data_list)):
		img = img_data_list[i]
		alphe_scale = round(random.random(), 2)
		gamma_bias = random.randint(-25, 25)

		Contrastimg = cv2.addWeighted(img, alphe_scale, img, 0.5, gamma_bias)

		save_count += 1
		save_name = "avg_"
		for n in range(6 - len(str(save_count))):
			save_name += "0"
		save_name += (str(save_count) + ".png")
		cv2.imwrite(save_path + "/" + save_name, Contrastimg)

		print("count: " + str(save_count) + " process: "
			+ str(i + 1) + "/" + str(len(img_data_list)))


		brightness = cv2.addWeighted(img, 0.5, img, 0.5, 40)


	return save_count

def getFlipAddList(img_data_list, save_path, save_count = 0):
	if do_shuffle:
		random.shuffle(img_data_list)

	H, W, C = img_data_list[0].shape

	for i in range(len(img_data_list)):
		img =  img_data_list[i]
		imgH = cv2.flip(img, 1)
		imgV = cv2.flip(img, 0)
		imgHV = cv2.flip(img, -1)

		img = img / 4 + imgH / 4 + imgV / 4 + imgHV / 4

		save_count += 1
		save_name = "avg_"
		for n in range(6 - len(str(save_count))):
			save_name += "0"
		save_name += (str(save_count) + ".png")
		cv2.imwrite(save_path + "/" + save_name, img)

		print("count: " + str(save_count) + " process: "
			+ str(i + 1) + "/" + str(len(img_data_list)))

	return save_count

if os.path.exists(to_path):
	shutil.rmtree(to_path)
os.makedirs(to_path)

if os.path.exists(from_path):
	file_list = os.listdir(from_path)
	for per_file in file_list:
		os.makedirs(to_path + per_file)
		img_list = os.listdir(from_path + per_file)

		img_data_list = []
		for n in range(len(img_list)):
			img_name = img_list[n]
			img = cv2.imread(from_path + per_file + "/" + img_name)
			img_data_list.append(img)
			print(from_path + per_file 
				+ " load process: " + str(n + 1) + "/" + str(len(img_list)))
		
		save_count = 0
		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 1)

		# save_count = getAvgList(img_data_list, to_path + per_file, 
		# 		save_count = save_count, merge_nums = 2)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 4)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 8)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 16)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 32)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 64)
		
		# save_count = getAvgList(img_data_list, to_path + per_file, 
		# 		save_count = save_count, merge_nums = 128)
		
		# save_count = getAvgList(img_data_list, to_path + per_file, 
		# 		save_count = save_count, merge_nums = 256)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 4)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 8)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 16)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 32)

		save_count = getAvgList(img_data_list, to_path + per_file, 
				save_count = save_count, merge_nums = 64)
		
		# save_count = getAvgList(img_data_list, to_path + per_file, 
		# 		save_count = save_count, merge_nums = 128)
		
		# save_count = getAvgList(img_data_list, to_path + per_file, 
		# 		save_count = save_count, merge_nums = len(img_list) - 1)

		# save_count = getFlipAddList(img_data_list, to_path + per_file, 
		# 		save_count = save_count)

		# save_count = getContrastAndBrightList(img_data_list, to_path + per_file, 
		# 		save_count = save_count)