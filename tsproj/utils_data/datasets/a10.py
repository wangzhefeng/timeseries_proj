# -*- coding: utf-8 -*-


# ***************************************************
# * File        : a10_data.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-15
# * Version     : 0.1.111520
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

import pandas as pd
from data_loader import read_ts


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DIRNAME = os.path.dirname(os.path.abspath(__file__))


class A10:

    def __init__(self):
        pass

    @staticmethod
    def get_local():
        ts_df = read_ts(
            filepath = os.path.join(DIRNAME, "data/a10.csv"),
            date_col = "date",
            date_format = '%Y-%m-%d',
            index_col = None,
            date_str = None,
            log = False,
        )

        return ts_df

    @staticmethod
    def get_github():
        ts_df = pd.read_csv(
            "https://raw.githubusercontent.com/selva86/datasets/master/a10.csv", 
            parse_dates = ['date'],
        )

        return ts_df


a10 = A10().get_local()


__all__ = [
    a10,
]




# 测试代码 main 函数
def main():
    a10 = a10().get_local()
    print(a10.head())

if __name__ == "__main__":
    main()

