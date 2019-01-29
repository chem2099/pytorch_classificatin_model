#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:35:01 2019

@author: wxz
"""
import numpy as np
import os
import time


class YOLO_Kmeans:
    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]  # 计算框的面积
        box_area = box_area.repeat(k)  # 每个数字重复9次
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]  # 计算初始聚类框的面积
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))  #
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)  # 取较小的值
        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)
        inter_area = inter_area.astype(np.float64)  # py2需要注意因为下一个运算为除法
        result = inter_area / (box_area + cluster_area - inter_area)  # 计算一个box与九个anchor的交并比
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))  # size = (box_number, ) 初始化每个框属于哪一个类
        np.random.seed()  # 初始化随机种子
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters 从boxs中随机挑选k个box作为初始聚类点
        while True:

            distances = 1 - self.iou(boxes, clusters)  # iou越大 distances越小

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters 找出某一类各个坐标的中位数作为新的聚类中心，比如输入第1
                    boxes[current_nearest == cluster], axis=0)  # 类的框有9个那么就从这九个框的宽中选出中位数，从这个
                # 9个框的高中选出中位数，然后将选的宽高作为新的聚类中心
            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            image_path, infos = line.strip().split(',')
            if infos == '':
                continue
            positions = infos.split(';')
            for position in positions:
                width = float(position.split('_')[2])
                height = float(position.split('_')[3])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]  # result.T[0, None]只取宽度，然后对result进行转置，然后按照
        self.result2txt(result)  # 宽度大小对result数据进行排序
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(  # 计算平均iou
            self.avg_iou(all_boxes, result) * 100))


'''
#适用于标注左上与右下坐标的数据
    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result
'''


def mkSubFile(lines, srcName, sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename = des_filename + '_' + str(sub) + extname
    print('make file: %s' % filename)
    fout = open(filename, 'w')
    try:
        #fout.writelines([head])
        fout.writelines(str(lines) + "\n")
        return sub + 1
    finally:
        fout.close()


def splitByLineCount(filename, count):
    with open(filename, 'r', encoding='UTF-8') as file:
        buf = []
        for line in file:
            line = line.strip('\n').split(',')
            buf.append(line[0])
            # if len(buf) == count:
            # buf = []
        f = open("train_1_w.txt", 'w')
        new_buf = list(set(buf))
        for i in range(len(new_buf)):
            writeline = str(new_buf[i]) + "\n"
            f.write(writeline)
        f.close()


if __name__ == "__main__":
    cluster_number = 15
    filename = "train_1w.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()

    '''链式seq
    from functional import seq
    from collections import Counter
    from operator import itemgetter

    # train_val = set(seq.open(filename, encoding='UTF-8'))
    #print(train_val)
    # total_val = sorted(filename.strip().split(',').count_by_value(), key=lambda x: x[1])
    '''

    '''文件读取写入的
    begin = time.time()
    splitByLineCount(filename, 20)
    end = time.time()
    print('time is %d seconds ' % (end - begin))
    '''