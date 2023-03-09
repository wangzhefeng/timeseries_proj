# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo.py
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


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def func():
    pass


class DemoClass:
    """
    类说明文档
    """
    _class_config_param = None  # 类私有不变量
    
    def __init__(self, id_):
        self.id = id_
        self.param_a = None  # 类公开变量
        self._internal_param = None  # 类私有变量
    
    def ClassDemoFunc(self):
        """
        类普通方法
        """
        pass
    
    def _ClassPrivateFunc(self):
        """
        类私有方法
        """
        pass


class _PrivateDemoClass:
    """
    私有类
    """
    
    def __init__(self):
        pass




# 测试代码 main 函数
def main():
    import pmdarima as pm

    x = pm.c(1, 2, 3, 4, 5, 6, 7)
    print(x)

    acf = pm.acf(x)
    print(acf)

    pm.plot_acf(x)

if __name__ == "__main__":
    main()

