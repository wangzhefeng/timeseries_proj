# -*- coding: utf-8 -*-


# *********************************************
# * Author      :canping Chen
# * Email       : canping.chen@you-i.net
# * Date        : 2021.09.06
# * Description : 序列数据预处理类函数
# * Link        : 
# **********************************************

import numpy as np
import pysindy as ps
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def SeriesDataNormalize(series):
    """
    数据序列归一化函数, 受异常值影响

    Parameters: 
        series: np.array (n, m)
    
    Returns:
        scaler: 归一化对象
        normalized: 归一化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(series)
    normalized = scaler.transform(series)

    return scaler, normalized


def SeriesDataStandardScaler(series):
    """
    数据序列标准化函数, 不受异常值影响

    Parameters: 
        series: np.array (n, m)
    
    Returns:
        scaler: 标准化对象
        normalized: 标准化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(series)
    normalized = scaler.transform(series)

    return scaler, normalized