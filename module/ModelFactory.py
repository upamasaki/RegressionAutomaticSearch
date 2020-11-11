###########################
# model import
#
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.svm import SVR, LinearSVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor

class ModelFactory():
    def __init__(self):
        pass

    def _LinearRegression(self):
        return [linear_model.LinearRegression(), 'LinearRegression']

    def _Ridge(self):
        return [Ridge(), 'Ridge']

    def _Lasso(self):
        return [Lasso(), 'Lasso']

    def _ElasticNet(self):
        return [ElasticNet(), 'ElasticNet']

    def _HuberRegresso(self):
        return [HuberRegressor(), 'HuberRegresso']

    def _DecisionTreeRegressor(self, max_depth_list):
        models = []
        names  = []
        base_name = 'DecisionTreeRegressor'
        for max_depth in max_depth_list:
            models.append(DecisionTreeRegressor(max_depth=max_depth))
            names.append("{}_max_depth{}".format(base_name, max_depth))
        return [models, names]

    def _RandomForestRegressor(self, max_depth_list, n_estimators_list):
        models = []
        names  = []
        base_name = 'RandomForestRegressor'
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                models.append(RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators))
                names.append("{}_max_depth{}_n{}".format(base_name, max_depth, n_estimators))
        return [models, names]

    def _AdaBoostRegressor(self, n_estimators_list):
        models = []
        names  = []
        base_name = 'AdaBoostRegressor'
        for n_estimators in n_estimators_list:
            models.append(AdaBoostRegressor(n_estimators=n_estimators))
            names.append("{}_n{}".format(base_name, n_estimators))
        return [models, names]

    def _PLSRegression(self, n_components_list):
        models = []
        names  = []
        base_name = 'PLSRegression'
        for n_components in n_components_list:
            models.append(PLSRegression(n_components=n_components))
            names.append("{}_n{}".format(base_name, n_components))
        return [models, names]

    def _BaggingRegressor(self, n_estimators_list, base_estimator=SVR()):
        models = []
        names  = []
        base_name = 'BaggingRegressor'
        for n_estimators in n_estimators_list:
            models.append(BaggingRegressor(base_estimator=SVR(), n_estimators=n_estimators))
            names.append("{}_n{}".format(base_name, n_estimators))
        return [models, names]

    def model_import(self):
        models_names = [ self._LinearRegression(), 
                         self._HuberRegresso(),
                         self._ElasticNet(),
                         self._Lasso(),
                         self._PLSRegression(n_components_list=list(range(3, 10, 1))),
                         self._BaggingRegressor(n_estimators_list=list(range(5, 30, 2))),
                         self._DecisionTreeRegressor(max_depth_list=list(range(2, 30, 2))),
                         self._RandomForestRegressor(max_depth_list=list(range(2, 30, 2)), n_estimators_list=list(range(20, 200, 5))),
                         self._AdaBoostRegressor(n_estimators_list=list(range(20, 200, 20)))]
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