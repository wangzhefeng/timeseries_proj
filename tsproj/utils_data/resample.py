# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-10-07
# * Version     : 0.1.100716
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


t = np.linspace(0, 3, 20, endpoint = False)
y = np.cos(t ** 2)
print(t)
print(len(t))
print(y)
print(len(y))


re_y = signal.resample(y, 40)
t_new = np.linspace(0, 3, len(re_y), endpoint = False)
print(t_new)
print(len(t_new))
print(re_y)
print(len(re_y))

# plot
# plt.plot(t, y, "b*--", label = "raw")
# plt.plot(t_new, re_y, "r.--", label = "resample data")
# plt.legend()
# plt.show();




# 原始数据
t = np.linspace(0, 3, 40, endpoint = False)
y = np.cos(t ** 2)

# 降采样, 40 个点采 20 个点
re_y = signal.resample(y, 20)
t_new = np.linspace(0, 3, len(re_y), endpoint = False)

# plot
# plt.plot(t, y, "b*--", label = "raw data")
# plt.plot(t_new, re_y, "r.--", label = "resample data")
# plt.legend()
# plt.show()


# 原始数据
t = np.array([
    0, 2, 3, 3.5, 4.5, 
    5.2, 6, 6.3, 8, 9, 
    10, 11.2, 12.3, 12.9, 14.5,
    16, 17, 18, 19, 20]) / 10
x = np.sin(3 * t)

# 重采样为等间隔
t1 = np.linspace(0, 2, 20, endpoint = True)
re_x, re_t = signal.resample(x, 20, t = t1)

# plot
# plt.plot(t, x, 'b*--', label = 'raw data')
# plt.plot(re_t, re_x, 'g.--', label = 'resample data')
# plt.legend()
# plt.show()


# 原始数据
t = np.array([
    0, 2, 3, 3.5, 4.5, 
    5.2, 6, 6.3, 8, 9, 
    10, 11.2, 12.3, 12.9, 14.5, 
    16, 17, 18, 19, 20])/10
x = np.sin(3 * t)

# 重采样
t1 = np.linspace(0, 2, 20, endpoint = True)
re_x = signal.resample(x, 20)

# plt.plot(t, x, 'b*--', label = 'raw data')
# plt.plot(t1, re_x, 'r.--', label = 'resample data')
# plt.legend()
# plt.show()


# 原始数据
t = np.linspace(0, 3, 40, endpoint = False)
y = np.cos(t ** 2)

# 等间隔降采样，40 个点采 20 个点
re_y, re_t = signal.resample(y, 20, t = t)

# plot
plt.plot(t, y, "b*--", label = "raw data")
plt.plot(re_t, re_y, "g.--", label = "resample data")
plt.legend()
plt.show()



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

