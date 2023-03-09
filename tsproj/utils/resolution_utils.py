# -*- coding: utf-8 -*-


# ***************************************************
# * File        : resolution_utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-05-23
# * Version     : 0.1.052311
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


from typing import Tuple


# 时区
TIMEZONE = "Asia/Shanghai"

# read resolution
RESOLUTION_1d = "1d"
RESOLUTION_1h = "1h"
RESOLUTION_1min = "1min"
RESOLUTION_5s = "5s"
RESOLUTION_1s = "1s"

# read interval, 当前时间间隔往前取 interval second 的数据,取resolution聚合的数据
# realtime
INTERVAL_REALTIME_START = 5 * 60 + 10
INTERVAL_REALTIME_END = - 5
# 5min
INTERVAL_5MIN_START = 5 * 60 + 10
INTERVAL_5MIN_END = - 5
# 15min
INTERVAL_15MIN_START = 15 * 60
INTERVAL_15MIN_END = - (3 * 60)

# 1hour
INTERVAL_1H_START = 3 * 3600
INTERVAL_1H_END = - (3 * 3600)
# 16hour
INTERVAL_16H_START = 17 * 3600
INTERVAL_16H_END = - (1 * 3600)


# 1day
def Get1dInterval(current_hour: int, current_minute: int, current_second: int)->Tuple[int,int]:

    """输入当前小时,分钟,秒,返回当时1天的开始及结束时间间隔

    Args:
        current_hour (int): 当前小时\n
        current_minute (int): 当前分钟\n
        current_second (int): 当前秒\n

    Returns:
        Tuple[int,int]: 当前1天的开始时间间隔,当前1天的结束时间间隔
    """
    INTERVAL_1D_START = (24 + current_hour) * 3600 + current_minute * 60 + current_second
    INTERVAL_1D_END = (current_hour - 2) * 3600 + current_minute * 60 + current_second
    return INTERVAL_1D_START, INTERVAL_1D_END


# 1day check
def Get1dCheckoutInterval(current_hour: int, current_minute: int, current_second: int) -> Tuple[int,int]:
    """输入当前小时,分钟,秒,返回当时近1天的开始及结束时间间隔

    Args:
        current_hour (int): 当前小时\n
        current_minute (int): 当前分钟\n
        current_second (int): 当前秒\n

    Returns:
        Tuple[int,int]: 当前近1天的开始时间间隔,当前近1天的结束时间间隔
    """
    INTERVAL_1D_CHECK_START = (24 + current_hour) * 3600 + current_minute * 60 + current_second
    INTERVAL_1D_CHECK_END = -(5 * 60)
    return INTERVAL_1D_CHECK_START, INTERVAL_1D_CHECK_END


# 3day
def Get3dInterval(current_hour: int, current_minute: int, current_second: int) -> Tuple[int,int]:
    """输入当前小时,分钟,秒,返回当时近3天的开始及结束时间间隔

    Args:
        current_hour (int): 当前小时\n
        current_minute (int): 当前分钟\n
        current_second (int): 当前秒\n

    Returns:
        Tuple[int,int]: 当前3天的开始时间间隔,当前3天的结束时间间隔
    """
    INTERVAL_3D_START = (24 * 3 + current_hour) * 3600 + current_minute * 60 + current_second
    INTERVAL_3D_END = (current_hour - 2) * 3600 + current_minute * 60 + current_second
    return INTERVAL_3D_START, INTERVAL_3D_END


# 30day
def Get30dInterval(current_hour: int, current_minute: int, current_second: int) -> Tuple[int,int]:
    """输入当前小时,分钟,秒,返回当时近30天的开始及结束时间间隔

    Args:
        current_hour (int): 当前小时\n
        current_minute (int): 当前分钟\n
        current_second (int): 当前秒\n

    Returns:
        Tuple[int,int]: 当前近30天的开始时间间隔,当前近30天的结束时间间隔
    """
    INTERVAL_30D_START = (24 * 30 + current_hour) * 3600 + current_minute * 60 + current_second
    INTERVAL_30D_END = -5
    return INTERVAL_30D_START, INTERVAL_30D_END
