# -*- coding: utf-8 -*-


# ***************************************************
# * File        : stamp_processing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-04-01
# * Version     : 0.1.040117
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


from datetime import datetime

import pytz
from pytz import timezone


def timestamp2datetime(stamp: int, time_zone="Asia/Shanghai") -> datetime:
    """
    将 Unix 时间戳转换为指定时区的日期时间格式

    Args:
        stamp (Number): 需要转化的时间戳 \n
        time_zone (str, 可选): 时区. Defaults to "Asia/Shanghai". \n

    Returns:
        datetime: datetime格式
    """
    local_tz = pytz.timezone(time_zone)
    datetime_data = datetime \
        .utcfromtimestamp(int(stamp)) \
        .replace(tzinfo = pytz.utc) \
        .astimezone(local_tz)
    
    return datetime_data


def align_timestamp(timestamp: int, time_zone = "Asia/Shanghai", resolution = "5s") -> int:
    """
    Align the UTC `timestamp` to the `resolution` with
    respect to `time_zone`.
    同 digital_machine.runtime.default.timestamps中_align_timestamp

    Args:
        t (Number): 需校准时间戳 \n
        time_zone (str, optional): 时区. Defaults to "Asia/Shanghai". \n
        resolution (str, optional): 需调整聚合度. Defaults to "1s". \n

    Raises:
        ValueError: 聚合度输入不正确

    Returns:
        int: 对齐后的时间戳

    >>> _align_timestamp(1503497069, "America/Chicago", resolution="1s")
    1503497069
    >>> _align_timestamp(1503497069, "UTC", resolution="5s")
    1503497065
    >>> _align_timestamp(1503497069, "Europe/Moscow", resolution="10s")
    1503497060
    >>> _align_timestamp(1503497069, "Europe/London", resolution="15s")
    1503497055
    >>> _align_timestamp(1503497069, "Europe/London", resolution="15sec")
    1503497055
    >>> _align_timestamp(1503497069, "Asia/Shanghai", resolution="1min")
    1503497040
    >>> _align_timestamp(1503497069, "Africa/Cairo", resolution="5min")
    1503496800
    >>> _align_timestamp(1503497069, "Europe/Brussels", resolution="10min")
    1503496800
    >>> _align_timestamp(1503497069, "Asia/Jerusalem", resolution="15min")
    1503496800
    >>> _align_timestamp(1503497069, "Asia/Calcutta", resolution="1h")
    1503495000
    >>> _align_timestamp(1503497069, "America/New_York", resolution="1h")
    1503496800
    >>> _align_timestamp(1503497069, "America/Los_Angeles", resolution="12h")
    1503471600
    >>> _align_timestamp(1503497069, "Australia/Sydney", resolution="1d")
    1503496800
    """
    tz = timezone(time_zone)
    if resolution is None:
        return timestamp
    elif resolution == "1s":
        return int(timestamp)
    elif resolution == "5s":
        return int(timestamp / 5) * 5
    elif resolution == "10s":
        return int(timestamp / 10) * 10
    elif resolution == "15s":
        return int(timestamp / 15) * 15
    # elif resolution == "15sec":
    #     return int(timestamp / 15) * 15
    elif resolution == "1min":
        return int(timestamp / 60) * 60
    elif resolution == "5min":
        return int(timestamp / 300) * 300
    elif resolution == "10min":
        return int(timestamp / 600) * 600
    elif resolution == "15min":
        return int(timestamp / 900) * 900
    elif resolution == "1h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds())
    elif resolution == "6h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        dt = dt.replace(hour = int(dt.hour / 6) * 6)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "8h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        dt = dt.replace(hour = int(dt.hour / 8) * 8)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "12h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        dt = dt.replace(hour = int(dt.hour / 12) * 12)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "1d":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "1mo":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "1y":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(month = 1, day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    else:
        raise ValueError("Invalid resolution: %s" % resolution)




def main():
    datetime_data = timestamp2datetime(1591148504)
    print(datetime_data)


if __name__ == "__main__":    
    main()

