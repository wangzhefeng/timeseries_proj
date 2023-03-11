# -*- coding: utf-8 -*-


# ***************************************************
# * File        : auto_arima.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-13
# * Version     : 0.1.111300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pmdarima as pm
from pmdarima.datasets import load_wineind


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
wineind = load_wineind().astype(np.float64)

# 拟合 stepwise auto-ARIMA
stepwise_fit = pm.auto_arima(
    y = wineind,
    start_p = 1,
    start_q = 1,
    max_p = 3,
    max_q = 3,
    seasonal = True,
    d = 1,
    D = 1,
    trace = True,
    error_action = "ignore",
    suppress_warnings = True,  # 收敛信息
    stepwise= True,
)

# 查看模型信息
print(pm.show_versions())
print(stepwise_fit.summary())


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

