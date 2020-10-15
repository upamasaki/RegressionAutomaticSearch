###########################
# model import
#
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor


class ModelFactory():
    def __init__(self):
        pass

    def LinearRegression(self):
        return [linear_model.LinearRegression(), 'LinearRegression']

    def DecisionTreeRegressor(self, max_depth_list):
        models = []
        names  = []
        base_name = 'DecisionTreeRegressor'
        for max_depth in max_depth_list:
            models.append(DecisionTreeRegressor(max_depth=max_depth))
            names.append("{}_max_depth{}".format(base_name, max_depth))
        return [models, names]

    def RandomForestRegressor(self, max_depth_list, n_estimators_list):
        models = []
        names  = []
        base_name = 'RandomForestRegressor'
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                models.append(RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators))
                names.append("{}_max_depth{}_n{}".format(base_name, max_depth, n_estimators))
        return [models, names]

    def AdaBoostRegressor(self, n_estimators_list):
        models = []
        names  = []
        base_name = 'AdaBoostRegressor'
        for n_estimators in n_estimators_list:
            models.append(AdaBoostRegressor(n_estimators=n_estimators))
            names.append("{}_n{}".format(base_name, n_estimators))
        return [models, names]

    def model_import(self):
        models_names = [ self.LinearRegression(), 
                         self.DecisionTreeRegressor(list(range(2, 30, 2))),
                         self.RandomForestRegressor(list(range(2, 30, 2)), list(range(20, 200, 20))),
                         self.AdaBoostRegressor(list(range(20, 200, 20)))]
        models = []
        names  = []
        for model_, name_ in models_names:
            if isinstance(model_, list):
                models.extend(model_)
                names.extend(name_)
            else:
                models.append(model_)
                names.append(name_)
        return models, names 


if __name__ == "__main__":

    model_factory = ModelFactory()
    models, names = model_factory.model_import()
    print("models : {}".format(models))
    print("names  : {}".format(names))
    # # model setting
    # models = [linear_model.LinearRegression(), 
    #           DecisionTreeRegressor(max_depth=5),
    #           RandomForestRegressor(max_depth=2, random_state=0),
    #           AdaBoostRegressor(random_state=0, n_estimators=100),
    #           BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)]
    # print(models)

    # names = ["LinearRegression", 
    #          "DecisionTreeRegressor",
    #          "RandomForestRegressor",
    #          "AdaBoostRegressor",
    #          "BaggingRegressor"]