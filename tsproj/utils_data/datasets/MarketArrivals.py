# -*- coding: utf-8 -*-


# ***************************************************
# * File        : MarketArrivals.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-15
# * Version     : 0.1.111522
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



class MarketArrivals:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_local():
        ts_df = read_ts(
            filepath = os.path.join(DIRNAME, "data/MarketArrivals.csv"),
            date_col = "date",
        )
        
        return ts_df


market_arrivals = MarketArrivals().get_local()


__all__ = [
    market_arrivals,
]




# 测试代码 main 函数
def main():
    market_arrivals = MarketArrivals().get_local()
    print(market_arrivals)

if __name__ == "__main__":
    main()

