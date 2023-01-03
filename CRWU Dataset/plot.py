# coding=utf-8
# @Time : 2022/10/8 14:50
# @Author : XiaoDong
# @File : plot.py
# @Software : PyCharm
import matplotlib.pyplot as plt
import numpy as np


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
    x2 = np.random.uniform(-1, 2, n)
    # 纵坐标y2
    y2 = np.array([(max_anomaly_score_0 + min_anomaly_score_1) / 2] * n)
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title('Result Analysis')
    # 设置横坐标名称
    ax1.set_xlabel('label')
    # 设置纵坐标名称
    ax1.set_ylabel('anomaly_score')
    # 画散点图
    ax1.scatter(temp_x1_0, temp_y1_0, s=s, c='y', marker='v', alpha=0.8)
    ax1.scatter(temp_x1_1, temp_y1_1, s=s, c='g', marker='>', alpha=0.8)
    # 画直线图
    ax1.plot(x2, y2, c='r', ls='--')
    # 调整横坐标的上下界
    plt.xlim(xmax=2, xmin=-1)
    # 显示
    plt.show()




# 主模块
if __name__ == "__main__":
    # 运行
    draw_scatter(n=3124, s=40)
