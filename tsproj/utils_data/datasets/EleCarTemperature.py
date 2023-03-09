# -*- coding: utf-8 -*-


# ***************************************************
# * File        : solar_panels.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-15
# * Version     : 0.1.111523
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


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DIRNAME = os.path.dirname(os.path.abspath(__file__))


class EleCarTemperature:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get():
        train = pd.read_csv(
            os.path.join(DIRNAME, "data/ele_car_temp/train.csv"), 
            header = 0,
        )
        test = pd.read_csv(
            os.path.join(DIRNAME, "data/ele_car_temp/test.csv"),
            header = 0,
        )
        
        return train, test


train, test = EleCarTemperature.get()


__all__ = [
    train,
    test,
]




# 测试代码 main 函数
def main():
    train, test = EleCarTemperature.get()
    print(train)
    print(test)

if __name__ == "__main__":
    main()

