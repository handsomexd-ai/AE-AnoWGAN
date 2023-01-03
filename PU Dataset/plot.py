# coding=utf-8
# @Time : 2022/10/8 14:50
# @Author : XiaoDong
# @File : plot.py
# @Software : PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc

# label,img_distance,anomaly_score,z_distance

# 定义画散点图的函数
def draw_scatter(n, s):
    """
    :param n: 点的数量，整数
    :param s:点的大小，整数
    :return: None
    """

    data = np.loadtxt('results/score.csv', encoding='utf-8', delimiter=',')
    # 通过切片获取横坐标x1
    x1 = data[:, 0]
    # 通过切片获取纵坐标R
    y1 = data[:, 2]
    fpr,tpr,thr = roc_curve(x1,y1,drop_intermediate=False)
    fpr,tpr = [0] + list(fpr),[0] + list(tpr)
    plt.plot(fpr,tpr,color = '#87CEFA',label = 'AE-AnoWGAN = 0.98')
    # plt.plot(fpr,tpr,color = 'red',label = 'FastFlow = 0.90')
    # plt.plot(fpr,tpr,color = 'orange',label = 'padim = 0.89')
    # plt.plot(fpr,tpr,color = 'yellow',label = 'stfpm = 0.88')
    # plt.plot(fpr,tpr,color = 'green',label = 'reverse_distillation = 0.85')
    # plt.plot(fpr,tpr,color = '#87CEFA',label = 'ganomaly = 0.71')
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    print(auc(fpr,tpr))

    # print(max(y1[1:3]))

    label0_count = 0
    label1_count = 0
    anomaly_score_0 = []
    anomaly_score_1 = []
    temp_x1_0 = []
    temp_y1_0 = []
    temp_x1_1 = []
    temp_y1_1 = []
    for i in range(len(x1)):
        if x1[i] == 0:
            label0_count = label0_count + 1
            anomaly_score_0.append(y1[i])
            temp_x1_0.append(x1[i])
            temp_y1_0.append(y1[i])
        else:
            label1_count = label1_count + 1
            anomaly_score_1.append(y1[i])
            temp_x1_1.append(x1[i])
            temp_y1_1.append(y1[i])
    print("label0_count:", label0_count)
    print("label1_count:", label1_count)
    max_anomaly_score_0 = max(anomaly_score_0)
    min_anomaly_score_1 = min(anomaly_score_1)
    print("max_anomaly_score_0:", max_anomaly_score_0)
    print("min_anomaly_score_1:", min_anomaly_score_1)
    print("anomaly_threshold:", ((max_anomaly_score_0 + min_anomaly_score_1) / 2))
    predict = []
    true_count = 0
    threshold = 0.023
    for i in range(len(x1)):
        if (x1[i] == 0 and y1[i] < threshold) or (
                x1[i] == 1 and y1[i] > threshold):
            predict.append("True")
            true_count = true_count + 1
        else:
            predict.append("False")
    print("predict_count:", len(predict))
    print("accuracy:", true_count / len(predict))
    # 横坐标x2
    x2 = np.random.uniform(0, 5000, n)
    # 纵坐标y2
    y2 = np.array([threshold] * n)
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title('Result Analysis')
    # 设置横坐标名称
    ax1.set_xlabel('image')
    # 设置纵坐标名称
    ax1.set_ylabel('anomaly_score')
    ax1.scatter(data[:len(temp_x1_0),4], temp_y1_0, s=s, c='#016392', marker='v', alpha=0.5,label = 'normal')
    ax1.scatter(data[len(temp_x1_0):,4], temp_y1_1, s=s, c='#FF6347', marker='o', alpha=0.3,label = 'anomaly')
    # 画直线图
    ax1.plot(x2, y2, c='r', ls='--',label = 'threshold')
    # 调整横坐标的上下界
    plt.legend()
    plt.xlim(xmax=5000, xmin=0)
    plt.show()

    data1 = np.loadtxt('results/score_shuffle.csv', encoding='utf-8', delimiter=',')
    fig1 = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax2 = fig1.add_subplot(1, 1, 1)
    # 设置标题
    ax2.set_title('PU Dataset Result Analysis')
    # 设置横坐标名称
    ax2.set_xlabel('Image Sequence')
    # 设置纵坐标名称
    ax2.set_ylabel('anomaly_score')
    temp1 = 0
    temp2 = 0
    for i in range(len(data1)):
        if data1[i,0] == 0:
            if temp1 == 0 :
                ax2.scatter(data1[i,4],data1[i,2],s=s, c='#016392', marker='v', alpha=0.8,label ='Normal')
                temp1 = 1
            else:
                ax2.scatter(data1[i, 4], data1[i, 2], s=s, c='#016392', marker='v', alpha=0.8)
        else:
            if temp2 == 0:
                ax2.scatter(data1[i,4],data1[i,2],s=s, c='#FF6347', marker='o', alpha=0.3,label = 'Anomaly')
                temp2 = 1
            else:
                ax2.scatter(data1[i, 4], data1[i, 2], s=s, c='#FF6347', marker='o', alpha=0.3)
    ax2.plot(x2, y2, c='r', ls='--', label='threshold')
    plt.xlim(xmax=5000, xmin=0)
    plt.legend()
    # ax1.scatter(data[:len(temp_x1_0),4], temp_y1_0, s=s, c='#016392', marker='v', alpha=0.5)
    # ax1.scatter(data[len(temp_x1_0):,4], temp_y1_1, s=s, c='g', marker='o', alpha=0.5)


    # 显示
    plt.show()









# 主模块
if __name__ == "__main__":
    # 运行
    draw_scatter(n=4899, s=40)
