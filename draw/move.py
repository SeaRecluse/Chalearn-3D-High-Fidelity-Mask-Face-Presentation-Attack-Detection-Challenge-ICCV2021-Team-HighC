import os
import shutil

if os.path.exists("../roc_p.txt"):
    if os.path.exists("./forTest/record/roc_p.txt"):
        os.remove("./forTest/record/roc_p.txt")
    shutil.copyfile("../roc_p.txt", "./forTest/record/roc_p.txt")

if os.path.exists("../roc_n.txt"):
    if os.path.exists("./forTest/record/roc_n.txt"):
        os.remove("./forTest/record/roc_n.txt")
    shutil.copyfile("../roc_n.txt", "./forTest/record/roc_n.txt")