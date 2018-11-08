import pandas as pd
from os import path


def out_put_data(Data):
    #进行数据的输出
    LocalPath = path.abspath(__file__)
    OutPutPath = path.abspath(path.join(LocalPath, "../../../data/submit_exmaple.csv"))
    Example = pd.read_csv(OutPutPath)
    OutPutIndex = Example["ID"]
    OutPut = pd.DataFrame(data={"ID": OutPutIndex, "value": Data})
    OutPut.to_csv(path.abspath("../../../data/Result/CCFResult.csv"), index=False)
