# -*- coding: utf-8 -*-


# ***************************************************
# * File        : StatsAnomalyDetection.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-18
# * Version     : 0.1.121808
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


"""
anomaly detection methods:

1. n_sigma: n sigma 判异
2. threshold: 异常阈值判异
3. iforest: 孤立森林
4. statistics: 统计量异常检测(均值、方差）
5. condition: 条件异常
6. KL: KL散度计算, 衡量两个分布的相似性
7. MK_MMD: 多核 MMD 计算, 衡量两个分布的相似性
8. correlation: 两个变量的相关性计算
9. correlation_ad: 两个变量相关性异常检测
"""


# python libraries
import os
import sys
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def NSigma(series, num_sigma: int = 3):
    """
    N Sigma Anomaly Detection

    Data should be normally distributed. 
    Under the 3-sigma principle, an outlier 
    that exceeds n standard deviations can 
    be regarded as an outlier  

    Args:
        series (_type_): pd.DataFrame with only one column or pd.Series
        num_sigma (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: the index list of outlier data
    """
    mu, std = series.mean(), series.std()
    lower, upper = mu - num_sigma * std, mu + num_sigma * std

    outlier_series = series[(lower > series) | (upper < series)]
    normal_series = series[(lower < series) & (upper > series)]
    labels = list(outlier_series.dropna().index)
    series = outlier_series.dropna()

    return labels, series


def Threshold(df, lower_limit, upper_limit):
    """
    Data greater than the upper limit or less 
    than the lower limit is considered as an outlier.

    Args:
        df (_type_): pd.DataFrame/pd.Series with only one column
        lower_limit (_type_): lower limit of the df value
        upper_limit (_type_): upper limit of the df value

    Returns:
        _type_: the index list of outlier data
    """
    outlier_df = df[(df > upper_limit) | (df < lower_limit)]
    labels = list(outlier_df.dropna().index)
    
    return labels


def IsolationForestModel(data, max_samples = "auto", contamination = "auto", threshold = None):
    """
    基于 isolation forest 算法的单特征时序数据异常值检测

    Args:
        data (_type_): _description_
        max_samples (str, optional): _description_. Defaults to "auto".
        contamination (str, optional): _description_. Defaults to "auto".
        threshold (_type_, optional): 异常值分数阈值. Defaults to None.

    Returns:
        _type_: 序列数据标签, -1 为异常值, 1 为非异常值
    
    Link:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = IsolationForest(max_samples=max_samples, contamination=contamination)
    if threshold is None:
        return clf.fit_predict(s_data)
    else:
        label_res = []
        scores = clf.score_samples(s_data)
        for score_i in scores:
            if score_i < threshold:
                label_res.append(-1)
            else:
                label_res.append(1)
        return label_res


def iforest(arr):
    """
    IsolationForest 
    - arr: 2D array
    - return: labels, scores, the index list of outlier data
    """
    iforest = IsolationForest()
    labels = iforest.fit_predict(arr)
    scores = iforest.decision_function(arr)
    outlier_index = np.argwhere(labels == -1).reshape(-1,)

    return labels, scores, outlier_index


def statistics(df, mean_min=None, mean_max=None, variance_max=None):
    """
    Statistics data(about 30 minutes) to judge if it's an anomaly.
    - df: should be pd.DataFrame with only one column
    - mean_min: minimum mean value
    - mean_max: maximum mean value
    - variance_max: maximum variance value
    - return: if the data abnormal. True-abnormal, false-normal
    """

    if mean_min is not None:
        if df.mean() < mean_min:
            return True
    if mean_max is not None:
        if df.mean() > mean_max:
            return True
    if variance_max is not None:
        if df.var() > variance_max:
            return True
    return False


def condition(condition_x, condition_y):
    """
    Need to difine condition_x and condition_y. When condition_x, if condition_y occurs, it's an anomaly.
    condition_x, condition_y should be a Boolean value. User
    """

    return condition_x and condition_y


def KL(y1, y2):
    """
    Calculate the similarity of two sets datas' distribution.
    - y1, y2: list
    - return: KL value, the smaller, the closer.
    """
    return scipy.stats.entropy(y1, y2) 


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    参考:https://www.zhihu.com/question/363994533/answer/2324371883
    多核或单核高斯核矩阵函数, 根据输入样本集x和y, 计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target: (b2, n)的Y分布样本数组
     kernel_mul: 多核MMD, 以bandwidth为中心, 两边扩展的基数, 比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定, 如果固定, 则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    """
     # 堆叠两组样本, 上面是X分布样本, 下面是Y分布样本, 得到(b1+b2,n）组总样本
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = np.concatenate((source, target), axis=0)
    # 对总样本变换格式为(1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上(b1+b2, b1+b2,n）, 相当于按行复制
    total0 = np.expand_dims(total,axis=0)
    total0= np.broadcast_to(total0,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # 对总样本变换格式为(b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上(b1+b2, b1+b2,n）, 相当于按复制
    total1 = np.expand_dims(total,axis=1)
    total1=np.broadcast_to(total1,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # total1 - total2 得到的矩阵中坐标(i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数, 对第三维进行求和, 即平方后再求和, 获得高斯核指数部分的分子, 是L2范数的平方
    L2_distance_square = np.cumsum(np.square(total0-total1),axis=2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    # 以fix_sigma为中值, 以kernel_mul为倍数取kernel_num个bandwidth值(比如fix_sigma为1时, 得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [np.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#多核合并


def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
     source: (b1,n)的X分布样本数组
     target:(b2, n)的Y分布样本数组
     kernel_mul: 多核MMD, 以bandwidth为中心, 两边扩展的基数, 比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定, 如果固定, 则为单核MMD
    Return:
     loss: MK-MMD loss
    """
    batch_size = int(source.shape[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分成4部分
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    # 这里计算出的n_loss是每个维度上的MK-MMD距离, 一般还会做均值化处理
    n_loss= loss / float(batch_size)
    return np.mean(n_loss)


def correlation(df, method="pearson"):
    """
    - method:pearson, kendall, spearman 对应着三种相关性计算方法
    默认method = pearson
    皮尔逊相关系数(pearson):连续性变量才可采用
    肯达相关系数(kendall):反映分类变量相关性的指标,适用于两个分类变量均为有序分类的情况。
    斯皮尔曼相关系数(spearman):利用两变量的秩次大小作线性相关分析,对原始变量的分布不作要求,属于非参数统计方法,适用范围要广些。
    
    - return:相关性系数, 越高越相关
    """
    return df.corr(method=method).iloc[1][0]


def correlation_ad(df, method="pearson", threshold=0.5):
    """
    根据事先离线分析出的两个变量直接的相关性大小, 设定阈值。一段时间内两个变量的相关性低于阈值, 
    认为存在异常
    - method:pearson, kendall, spearman
    - return:两个变量相关性是否异常。异常为True, 正常为False
    """
    return df.corr(method=method).iloc[1][0] < threshold


def ZScore(series: np.ndarray, threshold: float):
    """
    Z-score

    Args:
        series (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    z_score = (series - np.mean(series)) / np.std(series)

    return z_score


def BoxPlot(series: np.ndarray):
    """
    BoxPlot

    Args:
        series (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr + 1.5 * iqr

    return lower, upper


def IsolationForestOutlierDetection(data, max_samples="auto", contamination="auto", threshold=None):
    """
    基于isolation forest算法的单特征时序数据异常值检测
    Parameters:
        data: series list
        max_samples, contamination: 参考sklearn文档
        threshold: 异常值分数阈值
    Returns:
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = IsolationForest(max_samples=max_samples, contamination=contamination)
    if threshold is None:
        return clf.fit_predict(s_data)
    else:
        label_res = []
        scores = clf.score_samples(s_data)
        for score_i in scores:
            if score_i < threshold:
                label_res.append(-1)
            else:
                label_res.append(1)
        return label_res


def OneClassSvmOutlierDetection(data, kernel="brf", gamma=0.1, nu=0.3):
    """
    基于OneClassSvm算法的单特征时序数据异常值检测
    Parameters:
        data: series list
        kernel:
        threshold: 异常值分数阈值
    Returns:
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
    error_data = np.asarray(data).reshape(-1, 1)

    # fit the model
    clf = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    res = clf.fit_predict(error_data)
    return res


def LofOutlierDetection(data, neighbor=50, dist_metric="l1", contamination="auto"):
    """
    基于LOF算法的单特征时序数据异常值检测
    Parameters:
        data: series list
        neighbor: 近邻数
        dist_metric:距离计算方法
        contamination: 异常值比例
    Returns:
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = LOF(n_neighbors=neighbor, metric=dist_metric, contamination=contamination)
    res = clf.fit_predict(s_data)
    return res


# 测试代码 main 函数
def main():
    # ------------------------------
    # method test
    # ------------------------------
    data = pd.DataFrame({
        'metric': [1, 2, 3, 4, 3, 2, 4, 4, 8, 100, 7, 5, 3, 7, 9, 2]
    })
    # data = pd.Series({
    #     'metric': [1, 2, 3, 4, 3, 2, 4, 4, 8, 100, 7, 5, 3, 7, 9, 2]
    # })
    # test n_sigma
    labels, series = NSigma(series = data["metric"], num_sigma = 3)
    print(labels, series)
    
    # # test threshold
    # print(f"threshold outlier index = {threshold(data, 2, 10)}, value = {list(data['metric'].values[threshold(data, 2, 10)])}")
    # # test iforest
    # labels, scores, outlier_index = iforest(data["metric"].values.reshape(-1,1))
    # # print(f"iforest labels = {labels}, scores = {scores}")
    # print(f"iforest outlier index = {outlier_index}, value = {list(data['metric'].values[outlier_index])}")
    # # test KL
    # y1 = [np.random.randint(1,11) for i in range(10)]
    # y2 = [np.random.randint(1,11) for i in range(10)]
    # print("KL value =", KL(y1, y2))
    # # test MMD
    # y1 = np.array(y1).reshape(-1,1)
    # y2 = np.array(y2).reshape(-1,1)
    # print("MK_MMD loss =", MK_MMD(y1, y2))
    # # test correlation
    # corr_data = pd.DataFrame({"a": [1,2,3,4,5], "b": [1,3,5,4,2]})
    # print(f"correlation pearson result = {correlation(corr_data, method='pearson')},\
    #     kendall result = {correlation(corr_data, method='kendall')},\
    #     spearman result = {correlation(corr_data, method='spearman')}")
    # # test correlation anomaly detection
    # correlation_ad_result = correlation_ad(corr_data, method="pearson", threshold=0.5)
    # print(f"correlation_ad_result = {correlation_ad_result}")
    # ------------------------------
    # 
    # ------------------------------
    # print('\n异常数据如下: \n')
    # print(data.loc[labels])

    # plt.plot(data.index, data['销量'])
    # plt.plot(data.iloc[labels].index, data.iloc[labels][u'销量'], 'ro')
    # plt.show()

if __name__ == "__main__":
    main()

