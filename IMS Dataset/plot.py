# coding=utf-8
# @Time : 2022/10/8 14:50
# @Author : XiaoDong
# @File : plot.py
# @Software : PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve,auc

# label,img_distance,anomaly_score,z_distance

# 定义画散点图的函数
def draw_scatter(n, s):
    """
    :param n: 点的数量，整数
    :param s:点的大小，整数
    :return: None
    """
    # 加载数据
    data = np.loadtxt('results/score.csv', encoding='utf-8', delimiter=',')
    # 通过切片获取横坐标x1
    x1 = data[:, 0]
    # 通过切片获取纵坐标R
    y1 = data[:, 2]
    fpr,tpr,thr = roc_curve(x1,y1,drop_intermediate=False)
    fpr,tpr = [0] + list(fpr),[0] + list(tpr)
    plt.plot(fpr,tpr,color = '#87CEFA',label = 'TF-Anogan = 0.92')
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
    for i in range(len(x1)):
        if (x1[i] == 0 and y1[i] < ((max_anomaly_score_0 + min_anomaly_score_1) / 2)) or (
                x1[i] == 1 and y1[i] > ((max_anomaly_score_0 + min_anomaly_score_1) / 2)):
            predict.append("True")
            true_count = true_count + 1
        else:
            predict.append("False")
    print("predict_count:", len(predict))
    print("accuracy:", true_count / len(predict))
    # 横坐标x2
    # x2 = np.random.uniform(-1, 2, n)
    # # 纵坐标y2
    # y2 = np.array([(max_anomaly_score_0 + min_anomaly_score_1) / 2] * n)
    # # 创建画图窗口
    # fig = plt.figure()
    # # 将画图窗口分成1行1列，选择第一块区域作子图
    # ax1 = fig.add_subplot(1, 1, 1)
    # # 设置标题
    # ax1.set_title('Result Analysis')
    # # 设置横坐标名称
    # ax1.set_xlabel('label')
    # # 设置纵坐标名称
    # ax1.set_ylabel('anomaly_score')
    # # 画散点图
    # ax1.scatter(temp_x1_0, temp_y1_0, s=s, c='y', marker='v', alpha=0.8)
    # ax1.scatter(temp_x1_1, temp_y1_1, s=s, c='g', marker='>', alpha=0.8)
    # # 画直线图
    # ax1.plot(x2, y2, c='r', ls='--')
    # # 调整横坐标的上下界
    # plt.xlim(xmax=2, xmin=-1)
    # 显示
    # plt.show()

    data = np.loadtxt('results/score.csv', encoding='utf-8', delimiter=',')
    # 通过切片获取纵坐标R
    y1 = data[:1050,2]
    # for i in range(len(y1)):
    #     if y1[i] < 0.008 and i > 3 :
    #         y1[i] = max(y1[i-3:i+3])
    x_axis_data = range(1050)
    y_axis_data = y1
    # plt.plot(x_axis_data, y_axis_data, color='r',label = 'anomaly score')
    # plt.xlabel('image')
    # plt.ylabel('anomaly score')
    # plt.title('anomaly variation')
    # plt.show()
    plt.figure(dpi=110)
    data = pd.read_csv('results/score1.csv')
    g = sns.regplot(x='x', y='y', data=data,
                    marker='*',
                    order=10,  # 默认为1，越大越弯曲
                    scatter_kws={'s': 60, 'color': '#016392' },  # 设置散点属性，参考plt.scatter
                    line_kws={'linestyle': '--', 'color': '#FF6347'},
                    ci = 100,# 设置线属性，参考 plt.plot
                    )
    plt.xlabel('Image Sequence')
    plt.ylabel('anomaly score')
    plt.title('IMS Bearing Degradation')
    plt.show()

# 主模块
if __name__ == "__main__":
    # 运行
    draw_scatter(n=3124, s=40)
