import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import random

minShuffle = False
# minShuffle = True
loopC = 10000
limit_score = 1

upperBounds_th = 1.0
lowerBounds_th = 0.0

upperBounds_TPR = 1.01
lowerBounds_TPR = 0.8
tabFPR_list = [1e-5, 1e-3, 1e-2, 1e-1]
# tabFPR_list = [1e-10, 1e-4, 1e-3, 1e-2]
# tabFPR_list = [1e-10, 1e-5, 1e-4, 1e-3]

upperBounds_FRR = 0.01
lowerBounds_FRR = 0.00

line_density = 10

def to_percent(temp, position):
    return str(round((100*temp),3)) + '%'

def be_percent(temp):
    return str(round((100*temp),3)) + '%'

def getData(P_Name,N_Name):
    trueCompareList = []
    falseCompareList = []
    with open(P_Name,"r") as f:
        scoreList = f.readlines()
        for score in scoreList:
            if score:
                score = score.replace("\n","")
                score = score.split(" ")
                if len(score[1]) == 0:
                    continue
                elif score[1] == "-nan(ind)":
                    score[1] = 0
                # score = round(float(score[1]), 4)
                score = float(score[1])
                if score <= limit_score:
                    trueCompareList.append(score)

    with open(N_Name,"r") as f:
        scoreList = f.readlines()
        for score in scoreList:
            if score:
                score = score.replace("\n","")
                score = score.split(" ")
                if len(score[1]) == 0:
                    continue
                elif score[1] == "-nan(ind)":
                    score[1] = 0
                # score = round(float(score[1]), 4)
                score = float(score[1])
                if score <= limit_score:
                    falseCompareList.append(score)

    return trueCompareList, falseCompareList


def draw_TPR_FPR(filename, th_list, TPR_list, FPR_list, tpr_fpr_list, tpr_fpr_info_list):
    plt.subplot(1,2,1)
    plt.title("TPR-FPR")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(tabFPR_list[0] * 0.99, tabFPR_list[-1] * 1.01) #默认FPR: 0 ~ 1.0%
    plt.ylim(lowerBounds_TPR, upperBounds_TPR) #默认TPR: 50.0 ~ 100.0%
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    w = tabFPR_list[-1] * 1.01 - tabFPR_list[0] * 0.99
    h = upperBounds_TPR - lowerBounds_TPR
    for i in range(1,line_density + 1):
        plt.plot(
            [tabFPR_list[0] * 0.99 + w / line_density * i,
                tabFPR_list[0] * 0.99 + w / line_density * i],
            [lowerBounds_TPR,
                upperBounds_TPR],
            'k--',linewidth=1,color = "pink")
        plt.plot(
            [tabFPR_list[0] * 0.99,
                tabFPR_list[-1] * 1.01],
            [lowerBounds_TPR + h / line_density * i,
                lowerBounds_TPR + h / line_density * i],
            'k--',linewidth=1,color = "pink")

    str_res = ""
    tab_id = len(tabFPR_list) - 1
    for n in range(1, len(FPR_list)):
        fpr_score = FPR_list[n]
        if tab_id >= 0 and fpr_score < tabFPR_list[tab_id]:
            FPR_str = str(round(FPR_list[n - 1] * 100, 3)) + "%"
            TPR_str = str(round(TPR_list[n - 1] * 100, 3)) + "%"
            str_res += ("(FPR: " + FPR_str + ",TPR: " + TPR_str + ",TH: " + str(th_list[n - 1]) + ")\n")
            tab_id -= 1

        if FPR_list[n] == 0 and TPR_list[n] == 0:
            str_res += ("(FPR: " + str(0.00) + "%,TPR: " + str(0.00) + "%,TH: " + str(th_list[n]) + ")\n")
            break

    p, = plt.plot(FPR_list, TPR_list,'-',linewidth=1)
    tpr_fpr_list.append(p)
    tpr_fpr_info_list.append(filename + ":\n" + str_res)

    return tpr_fpr_list, tpr_fpr_info_list


def draw_FRR_FAR(filename, th_list, FRR_list, FAR_list, frr_far_list, frr_far_info_list):
    plt.subplot(1,2,2)
    plt.title("FRR-FAR")

    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.xlim(th_list[0] * 0.99, th_list[-1] * 1.01)
    plt.ylim(lowerBounds_FRR, upperBounds_FRR)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    w =  th_list[-1] * 1.01 - th_list[0] * 0.99
    h = upperBounds_FRR - lowerBounds_FRR
    for i in range(1,line_density + 1):
        plt.plot(
            [th_list[0] * 0.99 + w / line_density * i, 
                th_list[0] * 0.99 + w / line_density * i],
            [lowerBounds_FRR, 
                upperBounds_FRR],
            'k--',linewidth=1,color = "pink")
        plt.plot(
            [th_list[0] * 0.99, 
                th_list[-1] * 1.01],
            [lowerBounds_FRR + h / line_density * i,
                lowerBounds_FRR + h / line_density * i],
            'k--',linewidth=1,color = "pink")


    str_res = ""
    equal_idx = 0
    FAR_zero_min_idx = 0
    FRR_one_min_idx = 0
    alpha_count = 0
    beta_count = 1
    for n in range(1, len(th_list)):
        if FRR_list[n] == FAR_list[n]:
            equal_idx = n
        elif FRR_list[n] < FAR_list[n - 1] and FAR_list[n] > FRR_list[n - 1]:
            equal_idx = n

        if equal_idx != 0:
            beta_count += 1

        if FAR_zero_min_idx == 0:
            if FAR_list[n] / 1.0 == 0:
                FAR_zero_min_idx = n
        else:
            alpha_count += 1

        if FRR_one_min_idx == 0:
            if FRR_list[n] / 1.0 == 1.0:
                FRR_one_min_idx = n


    show_str = "FAR: " + str(round(FAR_list[equal_idx] * 100, 3)) + "%," \
        + "FRR: " + str(round(FRR_list[equal_idx] * 100, 3)) + "%," \
        + "TH: " + str(th_list[equal_idx]) + "\n"
    show_str += "FAR: " + str(0.0) + "%," \
        + "FRR: " + str(round(FRR_list[FAR_zero_min_idx] * 100, 3)) + "%," \
        + "TH: " + str(th_list[FAR_zero_min_idx]) + "\n"
    show_str += "FAR: " + str(0.0) + "%," \
        + "FRR: " + str(100.0) + "%," \
        + "TH: " + str(th_list[FRR_one_min_idx]) + "\n"

    w_beta_alpha = str(round( (1 - FRR_list[equal_idx]) * alpha_count / beta_count * 100.0, 3)) + "%\n"
    # print("score > " + str(th_list[equal_idx]) + " and > " + str(th_list[FAR_zero_min_idx]) + ": " + w_beta_alpha)
    # show_str += ("score > " + str(th_list[equal_idx]) + " and > " + str(th_list[FAR_zero_min_idx]) + ": " + w_beta_alpha)

    plt.scatter(th_list[equal_idx], FRR_list[equal_idx])
    plt.scatter(th_list[FAR_zero_min_idx], FRR_list[FAR_zero_min_idx])
    p, = plt.plot(th_list, FRR_list,'-',linewidth=1)
    frr_far_list.append(p)
    frr_far_info_list.append(filename)

    p, = plt.plot(th_list, FAR_list,'-',linewidth=1)
    frr_far_list.append(p)
    frr_far_info_list.append(show_str)

    FRR_list.sort()
    print(str(FRR_list[0] * 100) + "%")
    return frr_far_list, frr_far_info_list

def drawAll(filename, P_Name,N_Name, tpr_fpr_list, tpr_fpr_info_list, frr_far_list, frr_far_info_list):
    trueCompareList, falseCompareList = getData(P_Name,N_Name)
    
    if minShuffle:
        minLen = min(tsLen, fsLen)
        random.shuffle(trueCompareList)
        random.shuffle(falseCompareList)
        trueCompareList = trueCompareList[ : minLen]
        falseCompareList = falseCompareList[ : minLen]

    trueCompareList.sort()
    falseCompareList.sort()

    tsLen = len(trueCompareList)
    fsLen = len(falseCompareList)

    p_list = []
    coordinate_list = []
    th_list = []
    TPR_list = []
    FPR_list = []
    FRR_list = []
    FAR_list = []

    begin = (int)(loopC * lowerBounds_th)
    end =  (int)(loopC * upperBounds_th) + 1

    for threshold in range(begin, end):
        threshold = float(threshold/loopC)
        ts_count = 0
        fs_count = 0
        
        for ts in trueCompareList:
            if ts >= threshold:
                ts_count += 1

        for fs in falseCompareList:
            if fs >= threshold:
                fs_count += 1

        tpr_score = float(ts_count / tsLen)
        fpr_score = float(fs_count / fsLen)
        frr_score = 1 - tpr_score
        far_score = fpr_score

        th_list.append(threshold)
        TPR_list.append(tpr_score)
        FPR_list.append(fpr_score)
        FRR_list.append(frr_score)
        FAR_list.append(far_score)

        print("th: " + str(threshold) + ", FPR: " + str(fpr_score) + ", TPR: " + str(tpr_score) + ", FRR: " + str(frr_score) + ", FAR: " + str(far_score))

    tpr_fpr_list, tpr_fpr_info_list = draw_TPR_FPR(filename, th_list, TPR_list, FPR_list, 
        tpr_fpr_list, tpr_fpr_info_list)
    frr_far_list, frr_far_info_list = draw_FRR_FAR(filename, th_list, FRR_list, FAR_list, 
        frr_far_list, frr_far_info_list)

    return tpr_fpr_list, tpr_fpr_info_list, frr_far_list, frr_far_info_list

def getROC(record_path,p_list,coordinate_list):
    tpr_fpr_list = []
    tpr_fpr_info_list = []
    frr_far_list = []
    frr_far_info_list = []
    for filename in os.listdir(record_path):
        P_Name = record_path + filename + "/roc_p.txt"
        N_Name = record_path + filename + "/roc_n.txt"

        tpr_fpr_list, tpr_fpr_info_list, frr_far_list, frr_far_info_list = drawAll(filename, 
            P_Name,N_Name, tpr_fpr_list, tpr_fpr_info_list, frr_far_list, frr_far_info_list)


    return tpr_fpr_list, tpr_fpr_info_list, frr_far_list, frr_far_info_list


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
record_path_now = "./forTest/" #当前ROC测试的数据
show_r = 4 #百分比小数位数
plt.figure()
plt.rcParams['font.family'] = ['Times New Roman']
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

p_list = []
coordinate_list = []

# #绘制当前
record_path = record_path_now
p_tpr_fpr, p_tpr_fpr_infos, p_frr_far, p_frr_far_infos = getROC(record_path,p_list,coordinate_list)

plt.subplot(1,2,1)
plt.legend(p_tpr_fpr, p_tpr_fpr_infos, loc = 0, fontsize = "small")

plt.subplot(1,2,2)
plt.legend(p_frr_far, p_frr_far_infos, loc = 0, fontsize = "small")

plt.savefig("tmp.png", dpi=300)
plt.show()

