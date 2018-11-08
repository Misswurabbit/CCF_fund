from os import path
import pandas as pd


def loading_train_data():
    # 读入训练集表格
    LocalPath = path.abspath(__file__)
    DataPath = path.abspath(path.join(LocalPath, "../../../data/CCF"))
    TrainCorrPath = path.abspath(path.join(DataPath, "train_correlation.csv"))
    TrainFundPath = path.abspath(path.join(DataPath, "train_fund_return.csv"))
    TrainFundBenchmarkPath = path.abspath(path.join(DataPath, "train_fund_benchmark_return.csv"))
    TrainIndexPath = path.abspath(path.join(DataPath, "train_index_return.csv"))
    TrainCorr = pd.read_csv(TrainCorrPath)
    TrainFund = pd.read_csv(TrainFundPath)
    TrainBench = pd.read_csv(TrainFundBenchmarkPath)
    TrainIndex = pd.read_csv(TrainIndexPath, encoding="gbk")
    return TrainFund, TrainBench, TrainIndex, TrainCorr


def loading_test_data():
    # 读入测试集数据
    LocalPath = path.abspath(__file__)
    DataPath = path.abspath(path.join(LocalPath, "../../../data/CCF"))
    TestCorrPath = path.abspath(path.join(DataPath, "test_correlation.csv"))
    TestFundPath = path.abspath(path.join(DataPath, "test_fund_return.csv"))
    TestFundBenchmarkPath = path.abspath(path.join(DataPath, "test_fund_benchmark_return.csv"))
    TestIndexPath = path.abspath(path.join(DataPath, "test_index_return.csv"))
    TestCorr = pd.read_csv(TestCorrPath).reset_index(drop=True)
    TestFund = pd.read_csv(TestFundPath).reset_index(drop=True)
    TestBench = pd.read_csv(TestFundBenchmarkPath).reset_index(drop=True)
    TestIndex = pd.read_csv(TestIndexPath, encoding="gbk").reset_index(drop=True)
    return TestFund, TestBench, TestIndex, TestCorr


def train_data_reshape(TrainFund, TestFund, TrainBench, TestBench, TrainIndex, TestIndex,
                       TrainCorr, TestCorr, First, Last, CorrTime):
    # 对测试数据进行重新组合
    # 我知道很丑陋，下次再改
    TrainData = pd.DataFrame(
        data={"first_fund": TrainFund.iloc[First - 1, 1:], "first_fund_benchmark": TrainBench.iloc[First - 1, 1:],
              "last_fund": TrainFund.iloc[Last - 1, 1:],
              "last_fund_benchmark": TrainBench.iloc[Last - 1, 1:], "000300": TrainIndex.iloc[0, 1:],
              "000001": TrainIndex.iloc[1, 1:],
              "399001": TrainIndex.iloc[2, 1:], "399005": TrainIndex.iloc[3, 1:], "399006": TrainIndex.iloc[4, 1:],
              "000985": TrainIndex.iloc[5, 1:], "000016": TrainIndex.iloc[6, 1:], "000905": TrainIndex.iloc[7, 1:],
              "000010": TrainIndex.iloc[8, 1:], "000009": TrainIndex.iloc[9, 1:], "000903": TrainIndex.iloc[10, 1:],
              "000906": TrainIndex.iloc[11, 1:],
              "000852": TrainIndex.iloc[12, 1:], "000918": TrainIndex.iloc[13, 1:],
              "000919": TrainIndex.iloc[14, 1:],
              "H30351": TrainIndex.iloc[15, 1:], "H30352": TrainIndex.iloc[16, 1:],
              "000908": TrainIndex.iloc[17, 1:],
              "000909": TrainIndex.iloc[18, 1:],
              "000910": TrainIndex.iloc[19, 1:], "000911": TrainIndex.iloc[20, 1:],
              "000912": TrainIndex.iloc[21, 1:],
              "000913": TrainIndex.iloc[22, 1:], "000914": TrainIndex.iloc[23, 1:],
              "000915": TrainIndex.iloc[24, 1:],
              "000916": TrainIndex.iloc[25, 1:], "000917": TrainIndex.iloc[26, 1:],
              "HSI": TrainIndex.iloc[27, 1:],
              "H11001": TrainIndex.iloc[28, 1:], "000012": TrainIndex.iloc[29, 1:],
              "000013": TrainIndex.iloc[30, 1:],
              "000022": TrainIndex.iloc[31, 1:],
              "000832": TrainIndex.iloc[32, 1:], "000845": TrainIndex.iloc[33, 1:],
              "GF0001": TrainIndex.iloc[34, 1:],
              }).reset_index(drop=True)
    TrainTarget = pd.DataFrame(data={"correlation": TrainCorr.iloc[CorrTime, 1:]}).reset_index(drop=True)
    TestData = pd.DataFrame(
        data={"first_fund": TestFund.iloc[First - 1, 1:140], "first_fund_benchmark": TestBench.iloc[First - 1, 1:140],
              "last_fund": TestFund.iloc[Last - 1, 1:140],
              "last_fund_benchmark": TestBench.iloc[Last - 1, 1:140], "000300": TestIndex.iloc[0, 1:140],
              "000001": TestIndex.iloc[1, 1:140],
              "399001": TestIndex.iloc[2, 1:140], "399005": TestIndex.iloc[3, 1:140],
              "399006": TestIndex.iloc[4, 1:140],
              "000985": TestIndex.iloc[5, 1:140], "000016": TestIndex.iloc[6, 1:140],
              "000905": TestIndex.iloc[7, 1:140],
              "000010": TestIndex.iloc[8, 1:140], "000009": TestIndex.iloc[9, 1:140],
              "000903": TestIndex.iloc[10, 1:140],
              "000906": TestIndex.iloc[11, 1:140],
              "000852": TestIndex.iloc[12, 1:140], "000918": TestIndex.iloc[13, 1:140],
              "000919": TestIndex.iloc[14, 1:140],
              "H30351": TestIndex.iloc[15, 1:140], "H30352": TestIndex.iloc[16, 1:140],
              "000908": TestIndex.iloc[17, 1:140],
              "000909": TestIndex.iloc[18, 1:140],
              "000910": TestIndex.iloc[19, 1:140], "000911": TestIndex.iloc[20, 1:140],
              "000912": TestIndex.iloc[21, 1:140],
              "000913": TestIndex.iloc[22, 1:140], "000914": TestIndex.iloc[23, 1:140],
              "000915": TestIndex.iloc[24, 1:140],
              "000916": TestIndex.iloc[25, 1:140], "000917": TestIndex.iloc[26, 1:140],
              "HSI": TestIndex.iloc[27, 1:140],
              "H11001": TestIndex.iloc[28, 1:140], "000012": TestIndex.iloc[29, 1:140],
              "000013": TestIndex.iloc[30, 1:140],
              "000022": TestIndex.iloc[31, 1:140],
              "000832": TestIndex.iloc[32, 1:140], "000845": TestIndex.iloc[33, 1:140],
              "GF0001": TestIndex.iloc[34, 1:140],
              }).reset_index(drop=True)
    TestTarget = pd.DataFrame(data={"correlation": TestCorr.iloc[CorrTime, 1:]}).reset_index(drop=True)
    Data = TrainData.append(TestData).reset_index(drop=True)
    First = []
    Last = []
    for Num in range(0, 539):
        First.append(Data.iloc[Num, 0] - Data.iloc[Num, 2])
        Last.append(Data.iloc[Num, 1] - Data.iloc[Num, 3])
    Special = pd.DataFrame(data={"special_first": First, "special_last": Last})
    Data = pd.merge(Data, Special, left_index=True, right_index=True)
    Target = TrainTarget.append(TestTarget).reset_index(drop=True)
    return Data, Target


def test_data_reshape(Fund, Bench, Index, First, Last):
    # 对测试数据进行重新组合
    # 我知道很丑陋，下次再改
    TestData = pd.DataFrame(
        data={"first_fund": Fund.iloc[First - 1, -1], "first_fund_benchmark": Bench.iloc[First - 1, -1],
              "last_fund": Fund.iloc[Last - 1, -1],
              "last_fund_benchmark": Bench.iloc[Last - 1, -1], "000300": Index.iloc[0, -1],
              "000001": Index.iloc[1, -1], "399001": Index.iloc[2, -1], "399005": Index.iloc[3, -1],
              "399006": Index.iloc[4, -1], "000985": Index.iloc[5, -1], "000016": Index.iloc[6, -1],
              "000905": Index.iloc[7, -1], "000010": Index.iloc[8, -1], "000009": Index.iloc[9, -1],
              "000903": Index.iloc[10, -1], "000906": Index.iloc[11, -1], "000852": Index.iloc[12, -1],
              "000918": Index.iloc[13, -1], "000919": Index.iloc[14, -1], "H30351": Index.iloc[15, -1],
              "H30352": Index.iloc[16, -1], "000908": Index.iloc[17, -1], "000909": Index.iloc[18, -1],
              "000910": Index.iloc[19, -1], "000911": Index.iloc[20, -1], "000912": Index.iloc[21, -1],
              "000913": Index.iloc[22, -1], "000914": Index.iloc[23, -1], "000915": Index.iloc[24, -1],
              "000916": Index.iloc[25, -1], "000917": Index.iloc[26, -1], "HSI": Index.iloc[27, -1],
              "H11001": Index.iloc[28, -1], "000012": Index.iloc[29, -1], "000013": Index.iloc[30, -1],
              "000022": Index.iloc[31, -1], "000832": Index.iloc[32, -1], "000845": Index.iloc[33, -1],
              "GF0001": Index.iloc[34, -1]}, index=[0]).reset_index(drop=True)
    First = []
    Last = []
    First.append(TestData.iloc[0, 0] - TestData.iloc[0, 2])
    Last.append(TestData.iloc[0, 1] - TestData.iloc[0, 3])
    Special = pd.DataFrame(data={"special_first": First, "special_last": Last})
    TestData = pd.merge(TestData, Special, left_index=True, right_index=True)
    return TestData


def data_in(First, Last, CorrTime):
    # 进行输入数据的整理
    TrainFund, TrainBench, TrainIndex, TrainCorr = loading_train_data()
    TestFund, TestBench, TestIndex, TestCorr = loading_test_data()
    Data, Target = train_data_reshape(TrainFund, TestFund, TrainBench, TestBench, TrainIndex, TestIndex, TrainCorr,
                                      TestCorr, First, Last, CorrTime)
    Test = test_data_reshape(TestFund, TestBench, TestIndex, First, Last)
    Data = Data.astype(float)
    Target = Target.astype(float)
    Test = Test.astype(float)
    return Data, Target, Test
