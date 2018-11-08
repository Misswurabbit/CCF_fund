from os import path
import sys

sys.path.append(path.abspath(path.join(__file__, "../")))
from DataIn import data_in
from OutPut import out_put_data
from FeatureProcess import fea_process
from WzjStacking2 import Stacking2
from originalutils.WzjStacking import Stacking
from CCFutils.Stacking import stacking
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor


if __name__ == "__main__":
    Result = []
    CorrTime = 0
    for First in range(1, 201):
        for Last in range(First + 1, 201):
            Data, Target, Test = data_in(First, Last, CorrTime)
            Data, Test = fea_process(Data, Test)
            """
            model = RandomForestRegressor()
            model.fit(Data, Target)
            temp = model.predict(Data)
            print(explained_variance_score(temp,Target))
            """
            x_train, x_test, y_train, y_test = train_test_split(Data, Target, test_size=0.3, random_state=0)
            # Model = Stacking(FoldNum=5)
            # Model.fit(Data, Target)
            # Temp = Model.predict(Data)
            # Temp = float(stacking(Data, Test, Target, 10))
            # Result.append(Temp)
            CorrTime = CorrTime + 1
            print("XGBoost:       " + str(
                explained_variance_score(y_test, XGBRegressor().fit(x_train, y_train).predict(x_test))))
            print("LGBM:       " + str(
                explained_variance_score(y_test, LGBMRegressor().fit(x_train, y_train).predict(x_test))))
            print("RandomForest:       " + str(
                explained_variance_score(y_test, RandomForestRegressor().fit(x_train, y_train).predict(x_test))))

            #print("Stacking:      " + str(Stacking().score(x_train,y_train,x_test,y_test)))
            #print("Stacking2:     " + str(Stacking2().score(x_train, y_train, x_test, y_test)))
            print("first:" + str(First) + "  last:" + str(Last))
    out_put_data(Result)
    print("Over!")
