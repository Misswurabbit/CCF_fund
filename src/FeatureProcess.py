from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import math
import sys


def log_special(Num):
    # log运算
    Num = math.fabs(Num)
    if Num != 0:
        Result = math.log2(Num)
    else:
        Result = -sys.maxsize
    return Result


def fea_process(Train, Test):
    # 对数据进行一些初等数学运算以扩大数据特征的规模
    Index = Train.columns.values.tolist()
    # 获得列名
    for Row in Index:
        Temp = pd.DataFrame({Row + "sqr": [Num * Num for Num in Train[Row]],
                             Row + "sqrt": [math.sqrt(math.fabs(Num)) for Num in Train[Row]],
                             Row + "log": [log_special(num) for num in Train[Row]]
                             })
        Train = pd.merge(Train, Temp, right_index=True, left_index=True)
        Temp = pd.DataFrame({Row + "sqr": [Num * Num for Num in Test[Row]],
                             Row + "sqrt": [math.sqrt(math.fabs(Num)) for Num in Test[Row]],
                             Row + "log": [log_special(Num) for Num in Test[Row]]
                             })
        Test = pd.merge(Test, Temp, right_index=True, left_index=True)
    return Train, Test
