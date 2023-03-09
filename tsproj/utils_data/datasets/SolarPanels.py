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

from data_loader import read_ts


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DIRNAME = os.path.dirname(os.path.abspath(__file__))


class SolarPanels:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_metingen():
        df = read_ts(
            filepath = os.path.join(DIRNAME, "data/SolarPanels/metingen_27feb2022.csv"),
            date_col = 0,
        )

        return df

    @staticmethod
    def get_pv_elec_gas3():
        df = read_ts(
            filepath = os.path.join(DIRNAME, "data/SolarPanels/PV_Elec_Gas3.csv"),
            date_col = 0,
        )

        return df


solar_panels = SolarPanels()
metingen = solar_panels.get_metingen()
pv_elec_gas3 = solar_panels.get_pv_elec_gas3()


__all__ = [
    metingen,
    pv_elec_gas3,
]




# 测试代码 main 函数
def main():
    solar_panels = SolarPanels()
    
    metingen = solar_panels.get_metingen()
    pv_elec_gas3 = solar_panels.get_pv_elec_gas3()

    print(metingen)
    print(pv_elec_gas3)

if __name__ == "__main__":
    main()

