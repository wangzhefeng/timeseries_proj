# -*- coding: utf-8 -*-


# ***************************************************
# * File        : cfg_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-06-29
# * Version     : 0.1.062914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
import yaml


def load_yaml(file_name):
    with open(file_name, 'r', encoding = "utf-8") as infile:
        return yaml.load(
            infile, 
            Loader = yaml.FullLoader
        )


# --------------------
# 配置文件读取
# --------------------
cfg_dir = os.path.dirname(__file__)
# 项目配置 yaml 文件
sys_cfg_yaml = load_yaml(os.path.join(cfg_dir, "sys_cfg.yaml"))
# 项目后端配置 yaml 文件
api_cfg_yaml = load_yaml(os.path.join(cfg_dir, "backend_api_cfg.yaml"))
# --------------------
# 热电项目配置信息
# --------------------
# 热电平衡项目配置文件路径
project_path = sys_cfg_yaml["project_cfg_path"]
if project_path is not None:
    # 热电平衡项目项目名称
    project_name = project_path.split('.')[0]
    # 热电平衡项目日志名称
    project_log_name = project_name.split("_")[-1]
    # 热电平衡配置参数
    sys_cfg = load_yaml(os.path.join(cfg_dir, project_path))
    # 热电后端配置参数
    api_cfg = api_cfg_yaml[project_name]
else:
    project_name = None
    project_log_name = "TEP_BASE"
    sys_cfg = None
    api_cfg = None



# 测试代码 main 函数
def main(): 
    pass

if __name__ == "__main__":
    main()

