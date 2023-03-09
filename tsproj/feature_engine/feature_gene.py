# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071720
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


__all__ = [
    "gen_time_features",
    "get_time_fe",
]


# python libraries
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from utils_func import is_weekend


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def gen_time_features(ts: pd.DataFrame, dt_format: str, dt_is_index: bool = False, dt_name: str = None, features: List = None) -> pd.DataFrame:
    """
    时间特征提取

    Args:
        data ([type]): 时间序列
        datetime_format ([type]): 时间特征日期时间格式
        datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
        datetime_name ([type], optional): 时间特征名称. Defaults to None.
        target_name ([type], optional): 目标特征名称. Defaults to False.
        features: 最后返回的特征名称列表
    """
    # 数据拷贝
    data = ts.copy()
    # 日期时间特征处理
    if dt_is_index:
        # df = df.set_index("季度")
        # df.index = pd.to_datetime(df.index)
        data["DT"] = data.index
        data["DT"] = pd.to_datetime(data["DT"], format = dt_format)
    else:
        data[dt_name] = pd.to_datetime(data[dt_name], format = dt_format)
        data["DT"] = data[dt_name]
    # 时间特征提取
    data["year"] = data["DT"].apply(lambda x: x.year)  # 年
    # data["year"] = data["DT"].dt.year
    data["quarter"] = data["DT"].apply(lambda x: x.quarter)  # 季度
    # data["quarter"] = data["DT"].dt.quarter
    data["month"] = data["DT"].apply(lambda x: x.month)  # 月
    # data["month"] = data["DT"].dt.month
    data["day"] = data["DT"].apply(lambda x: x.day)  # 日
    # data["day"] = data["DT"].dt.day
    data["hour"] = data["DT"].apply(lambda x: x.hour)  # 时
    # data["hour"] = data["DT"].dt.hour
    data["minute"] = data["DT"].apply(lambda x: x.minute)  # 分
    # data['minute'] = data['DT'].dt.minute
    data["second"] = data["DT"].apply(lambda x: x.second)  # 秒
    # data['second'] = data['DT'].dt.second
    data["dayofweek"] = data["DT"].apply(lambda x: x.dayofweek)  # 一周的第几天
    # data["dayofweek"] = data["DT"].dt.day_of_week
    data["dayofyear"] = data["DT"].apply(lambda x: x.dayofyear)  # 一年的第几天
    # data["dayofyear"] = data["DT"].dt.day_of_year
    data["weekofyear"] = data["DT"].apply(lambda x: x.weekofyear)  # 一年的第几周
    # data["weekofyear"] = data["DT"].dt.week
    data["is_year_start"] = data["DT"].apply(lambda x: x.is_year_start)  # 是否年初
    # data["is_year_start"] = data["DT"].dt.is_year_start
    data["is_year_end"] = data["DT"].apply(lambda x: x.is_year_end)  # 是否年末
    # data["is_year_end"] = data["DT"].dt.is_year_end
    data["is_quarter_start"] = data["DT"].apply(lambda x: x.is_quarter_start)  # 是否季度初
    # data["is_quarter_start"] = data["DT"].dt.is_quarter_start
    data["is_quarter_end"] = data["DT"].apply(lambda x: x.is_quarter_end)  # 是否季度末
    # data["is_quarter_start"] = data["DT"].dt.is_quarter_end
    data["is_month_start"] = data["DT"].apply(lambda x: x.is_month_start)  # 是否月初
    # data["is_month_start"] = data["DT"].dt.is_month_start
    data["is_month_end"] = data["DT"].apply(lambda x: x.is_month_end)  # 是否月末
    # data["is_month_end"] = data["DT"].dt.is_month_end
    data["weekend"] = data['dow'].apply(is_weekend)  # 是否周末
    data["day_high"] = data["hour"].apply(lambda x: 0 if 0 < x < 8 else 1)  # 是否为高峰期
    # 删除临时日期时间特征
    del data["DT"]
    # 特征选择
    if features is None:
        result = data
    else:
        result = data[features]
    
    return result 


def get_time_fe(data: pd.DataFrame, col: str, n: int, one_hot: bool = False, drop: bool = True):
    """
    构造时间特征

    Args:
        data (_type_): _description_
        col (_type_): column name
        n (_type_): 时间周期
        one_hot (bool, optional): _description_. Defaults to False.
        drop (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    data[col + '_sin'] = round(np.sin(2 * np.pi / n * data[col]), 6)
    data[col + '_cos'] = round(np.cos(2 * np.pi / n * data[col]), 6)
    if one_hot:
        ohe = OneHotEncoder()
        X = ohe.fit_transform(data[col].values.reshape(-1, 1)).toarray()
        df = pd.DataFrame(
            X, 
            columns = [col + '_' + str(int(i)) for i in range(X.shape[1])]
        )
        data = pd.concat([data, df], axis = 1)
        if drop:
            data = data.drop(col, axis = 1)

    return data


def gen_statistic_features(ts: pd.DataFrame) -> pd.DataFrame:
    pass






# 测试代码 main 函数
def main():
    data = None
    data_df = gen_time_features(data)
    data_df = get_time_fe(data_df, 'hour', n = 24, one_hot = False, drop = False)
    data_df = get_time_fe(data_df, 'day', n = 31, one_hot = False, drop = True)
    data_df = get_time_fe(data_df, 'dayofweek', n = 7, one_hot = True, drop = True)
    data_df = get_time_fe(data_df, 'season', n = 4, one_hot = True, drop = True)
    data_df = get_time_fe(data_df, 'month', n = 12, one_hot = True, drop = True)
    data_df = get_time_fe(data_df, 'weekofyear', n = 53, one_hot = False, drop = True)

if __name__ == "__main__":
    main()

