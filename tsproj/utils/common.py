# -*- coding: utf-8 -*-


# ***************************************************
# * File        : common.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-07
# * Version     : 0.1.120722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import logging
import traceback
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd


def SafeArraySum(array_data: np.ndarray) -> np.float64:
    """
    针对 Numpy Array的安全加法.
    包括筛除 nan,以及排错
    
    Args:
        array_data (numpy.ndarray): Numpy Array

    Returns:
        当输入为 ndarray 全为 int 时,返回 numpy.int32 
        当输入中 ndarray 全为 float32 时,返回 numpy.float32 
        当输入中 ndarray 有 float64/floa t时,返回 numpy.float64 
        当遇到错误时,返回 numpy.nan 类型 float
        对空 array 进行计算时返回 0.0, 类型 float64
    """
    try:
        return np.nansum(array_data)
    except:
        traceback.print_exc()
        return np.nan


def SafeArrayAvg(array_data: np.ndarray) -> np.float64:
    """
    针对 Numpy Array的安全平均法.

    Args:
        array_data (numpy.ndarray): Numpy Array

    Returns:
        当正常输入输出,返回 numpy.float64 
        当输入中 ndarray 全为 float32 时,返回 numpy.float32 
        当遇到错误时,返回 numpy.nan 类型 float
        对空 array 进行计算时返回 numpy.nan,类型 float
    """
    try:
        return np.nanmean(array_data)
    except:
        traceback.print_exc()
        return np.nan


def GetColumnWithCheck(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    检查并获取 pd.DataFrame 列的值

    Args:
        df (pd.DataFrame): pandas DataFrame
        column (str): 需要获取的列名

    Returns:
        np.ndarray: 
        如果列名在此 DataFrame 中,返回该列的值,类型 np.ndarray.
        如果此列名不在此 DataFrame 中, 返回 None
    """
    if column in df.columns:
        return df[column].values
    else:
        return None


def GetEnumInfo(enum_cls, info):
    """
    获取枚举类型的
    """
    if enum_cls._value2member_map_.get(info) is not None:
        return enum_cls._value2member_map_[info]
    elif enum_cls._member_map_.get(info) is not None:
        return enum_cls._member_map_[info]
    else:
        return None


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


def CheckValid(value: Union[Number,np.ndarray,list,pd.DataFrame], _return: bool = True, **kwargs):
    """_summary_

    Args:
        value (Number,np.ndarray,list,pd.DataFrame): 数字或数组
        _return (bool, optional): 判断结果选择，默认判断真. Defaults to True.

    Returns:
        _type_: 是否有效
    """
    try:
        if is_number(value):
            #NOTE nan特殊处理
            if np.isnan(value):
                return not _return
            if "_min" in kwargs:
                if float(value)> kwargs["_min"]:
                    return _return
                else:
                    return not _return
            if "_max" in kwargs:
                if float(value)< kwargs["_max"]:
                    return _return
                else:
                    return not _return
            return _return
    except TypeError:
        pass
    try:
        # NOTE 增加可自定义参数用来控制
        if len(value) > 1:
            return _return
    except TypeError:
        pass
    return not _return


def GetValidValueFromPrimary(_domain_dict: dict, _metric: str, _instance_name: str, _default = np.nan):
    """
    _summary_

    Args:
        _domain_dict (dict): 属性数域
        _metric (str): 查询metric
        _instance_name (str): 当前查询实例名

    Returns:
        _type_: 正常取实际设定属性数域里的值,如果有默认需求可修改
    """
    if _metric in _domain_dict.keys():
        return _domain_dict[_metric]
    else:
        logging.error(f"{_instance_name} has no {_metric} set in primary domain set np.nan")
        return _default


def CheckValidBound(bound_limit: list):
    """
    检测边界条件是否合理
    Args:
        bound_limit: [min, max]
    Returns:
        True/False
    """
    if np.nan in bound_limit:
        logging.error(f"nan bound limit: {bound_limit}")
        return False
    
    if bound_limit[0] >= bound_limit[1]:
        logging.error(f"not valid limit: {bound_limit}")
        return False
    return True


class DotDict(dict):
    """
    dot.notation access to dictionary attributes

    Args:
        dict (_type_): _description_
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




if __name__ == "__main__":
    a = [1, np.nan, 3]
    print(SafeArraySum(a))
    assert CheckValid(1) == True
    assert CheckValid(None) == False
    assert CheckValid(np.nan) == False
    assert CheckValid([1,2,3]) == True
    assert CheckValid(np.array([1,2,3])) == True
    assert CheckValid(pd.DataFrame({"a":[1,2,3]})) == True
    assert CheckValid(1.1) == True
    assert CheckValid(1.1,_min=0.1) == True
    assert CheckValid(None,_min=0.1) == False
    assert CheckValid(np.nan,_min=0.1) == False
    assert CheckValid(0.01,_min=0.1) == False
    assert CheckValid(1.1,_max=0.1) == False
    assert CheckValid(None,_max=0.1) == False
    assert CheckValid(np.nan,_max=0.1) == False
    assert CheckValid(0.01,_max=0.1) == True
