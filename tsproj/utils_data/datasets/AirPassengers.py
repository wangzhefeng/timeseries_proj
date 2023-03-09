# -*- coding: utf-8 -*-


# ***************************************************
# * File        : air_passengers_data.py
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

from data_loader import read_ts


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DIRNAME = os.path.dirname(os.path.abspath(__file__))


class AirPassengers:

    def __init__(self):
        pass

    @staticmethod
    def get_local():
        ts_df = read_ts(
            filepath = os.path.join(DIRNAME, "data/AirPassengers.csv"),
            date_col = 0,
            date_format = "%Y-%m",
        )

        return ts_df

    @staticmethod
    def get_darts():
        from darts.datasets import AirPassengersDataset

        ts_df = AirPassengersDataset().load()

        return ts_df


air_passengers = AirPassengers().get_local()


__all__ = [
    air_passengers,
]




# 测试代码 main 函数
def main():
    air_passengers = AirPassengers().get_local()
    print(air_passengers)

if __name__ == "__main__":
    main()

