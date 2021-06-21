import os
import glob
import shutil
import random

import torch
import torch.nn as nn
from model import ShuffleNetV2
import torchvision.transforms as transforms
from PIL import Image
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_path = './model_best.pth.tar'
ONLY_VAL = 0

#===================================================================
load = "./orig_data/sort_val/"
real_txt = "./realList.txt"
fake_txt = "./fakeList.txt"

if os.path.exists(real_txt):
    os.remove(real_txt)

if os.path.exists(fake_txt):
    os.remove(fake_txt)

floders = glob.glob(load + "/*")

f_fake = open(fake_txt, "a", newline="\n", encoding="utf-8")
f_real = open(real_txt, "a", newline="\n", encoding="utf-8")
for floder in floders:
    print(floder)
    if "0" in floder:
        fake_faces = glob.glob(floder + "/*")
        for per_face in fake_faces:
            f_fake.write(per_face + "\n")
    elif "1" in floder:
        real_faces = glob.glob(floder + "/*")
        for per_face in real_faces:
            f_real.write(per_face + "\n")

f_fake.close()
f_real.close()
#===================================================================

def as_num(x):
     y='{:.5f}'.format(x) # 5f表示保留5位小数点的float型
     return y

INPUT_SIZE = 224
INPUT_CHNS = 3
c_tag = 2
stages_out_channels = [24, 48, 96, 192, 1024]
if c_tag == 1:
    stages_out_channels = [24, 116, 232, 464, 1024]
elif c_tag == 1.5:
    stages_out_channels = [24, 176, 352, 704, 1024]
elif c_tag == 2:
      stages_out_channels = [24, 244, 488, 976, 2048]

stages_repeats = [4, 8, 4]
 
def randScore(bias):
    return (random.random() + 1) / 4 + bias

model = ShuffleNetV2(stages_repeats = stages_repeats, stages_out_channels = stages_out_channels)
checkpoint = torch.load(model_path, map_location=device)

state_dict = model.state_dict()
for (k,v) in checkpoint['state_dict'].items():
    key = k[7:]
    print(key)
    if key in state_dict:
        state_dict[key] = v
    else:
        state_dict.update({key : v})

model.load_state_dict(state_dict)
model.to(device = device)
model.eval()

trans_test = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
            ])
softmax = nn.Softmax(dim=1)

def writeRes(load_path, save_path):
    if os.path.exists(load_path):
        line_list = []
        with open(load_path, "r") as f:
            line_list = f.readlines()
        
        save_txt = open(save_path, "w", newline = "\n")
        for n in range(len(line_list)):
            per_line = line_list[n].replace("\n", "")
            if per_line and os.path.exists(per_line):
                img = Image.open(per_line)
                input_tensor = trans_test(img).to(device = device)
                input_tensor = input_tensor.view(
                        1, INPUT_CHNS, INPUT_SIZE, INPUT_SIZE)

                output = softmax(model(input_tensor)).detach().cpu().numpy()[0][1]
                output = as_num(output)
                save_txt.write(str(n + 1) + " " + output + "\n")
                print(str(n + 1) + "/" + str(len(line_list)) + ": " + output)

def mergeRes(load_prehead, load_txt_path, save_txt_path):
    if os.path.exists(load_txt_path):
        save_txt = open(save_txt_path, "a", newline = "\n")

        with open(load_txt_path, "r") as f:
            line_list = f.readlines()
            for n in range(len(line_list)):
                per_line = line_list[n].replace("\n", "")
                if per_line:
                    img = Image.open(load_prehead + per_line)
                    input_tensor = trans_test(img).to(device = device)
                    input_tensor = input_tensor.view(
                            1, INPUT_CHNS, INPUT_SIZE, INPUT_SIZE)

                    output = softmax(model(input_tensor)).detach().cpu().numpy()[0][1]
                    output = as_num(output)
                    save_txt.write(per_line + " " + output + "\n")
                    print(str(n + 1) + "/" + str(len(line_list)) 
                        + "\t" + per_line + " " + output)
        save_txt.close()


start = time.time()
if ONLY_VAL:
    writeRes("./realList.txt", "roc_p.txt")
    writeRes("./fakeList.txt", "roc_n.txt")
else:
    save_txt_path = "./HighC.txt"
    if os.path.exists(save_txt_path):
        os.remove(save_txt_path)
    mergeRes("./orig_data/save_merges_", "./orig_data/val.txt", save_txt_path)
    mergeRes("./orig_data/save_merges_", "./orig_data/test.txt", save_txt_path)

print("cost time: " + str(round(time.time() - start, 4)) + " s")