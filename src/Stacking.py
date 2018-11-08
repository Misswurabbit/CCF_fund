from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost.sklearn import XGBRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def test_process(Test):
    # 对训练集进行标准化后列名变化，而xgboost对列名有要求（要求一致），此处对列名统一化
    Num = 0
    for col in Test:
        Test.rename(columns={col: Num}, inplace=True)
        Num += 1
    return Test


def get_name(obj, NameSpace):
    # 获得obj的name
    return [Name for Name in NameSpace if NameSpace[Name] is obj][0]


def stacking(Data, Test, Target, FoldNum):
    BaseModel = [RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), ExtraTreesRegressor(),
                 SVR()]
    EnsembleModel = XGBRegressor()
    # 模型初始化
    Scale = StandardScaler()
    Data = pd.DataFrame(Scale.fit_transform(Data))
    Test = test_process(Test)
    # 数据标准化和归一化
    BaseTrainFold = []
    BaseTestFold = []
    BaseTargetFold = []
    KF = KFold(n_splits=FoldNum)
    for TrainIndex, TestIndex in KF.split(Data):
        BaseTrainFold.append(Data.iloc[TrainIndex].reset_index(drop=True))
        BaseTestFold.append(Data.iloc[TestIndex].reset_index(drop=True))
        BaseTargetFold.append(Target.iloc[TrainIndex].reset_index(drop=True))
    # 针对BaseModel进行数据集的划分
    EnsembleTrainFold = []
    EnsembleTestFold = []
    Mark = 0
    for Model in BaseModel:
        Mark += 1
        TrainFold = []
        TestFold = []
        for Num in range(FoldNum):
            Clf = Model
            Clf.fit(BaseTrainFold[Num], BaseTargetFold[Num])
            TrainFold.append(pd.DataFrame(data={"data" + str(Mark): Clf.predict(BaseTestFold[Num])}))
            TestFold.append(pd.DataFrame(data={"data" + str(Mark): Clf.predict(Test)}))
            if Num == FoldNum - 1:
                TrainTemp = TrainFold[0]
                TestTemp = TestFold[0]
                for Index in range(1, FoldNum):
                    TrainTemp = TrainTemp.append(TrainFold[Index])
                    TestTemp = TestTemp.append(TestFold[Index])
                TrainTemp.reset_index(inplace=True, drop=True)
                TestTemp.reset_index(inplace=True, drop=True)
                EnsembleTrainFold.append(TrainTemp)
                EnsembleTestFold.append(TestTemp)
    EnsembleTrain = EnsembleTrainFold[0]
    EnsembleTest = EnsembleTestFold[0]
    for Index in range(1, len(EnsembleTrainFold)):
        EnsembleTrain = pd.merge(EnsembleTrain, EnsembleTrainFold[Index], left_index=True, right_index=True)
        EnsembleTest = pd.merge(EnsembleTest, EnsembleTestFold[Index], left_index=True, right_index=True)
    # 第一层模型进行数据拟合和处理
    EnsembleModel.fit(EnsembleTrain, Target)
    EnsembleResult = EnsembleModel.predict(EnsembleTest)
    Result = 0
    for Num in EnsembleResult:
        Result += Num
    Result = Result / len(EnsembleResult)
    # 对测试集结果求平均值进行输出
    return Result
