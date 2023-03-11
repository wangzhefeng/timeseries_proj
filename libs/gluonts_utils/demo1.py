# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo1.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-08
# * Version     : 0.1.120815
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx import DeepAREstimator, Trainer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# model
dataset = get_dataset("airpassengers")

# model
deepar = DeepAREstimator(
    prediction_length = 12,
    freq = "M",
    trainer = Trainer(epochs = 5),
)
model = deepar.train(dataset.train)

# model predict
true_values = to_pandas(list(dataset.test)[0])
true_values.to_timestamp().plot(color = "k")

prediction_input = PandasDataset([
    true_values[:-36],
    true_values[:-24],
    true_values[:-12],
])
predictions = model.predict(prediction_input)

# plotting
for color, prediction in zip(["green", "blue", "purple"], predictions):
    prediction.plot(color = f"tab:{color}")
plt.legend(["True values"], loc = "upper left", fontsize = "xx-large")
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

