from fedtree import FLRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
    x, y = load_svmlight_file("../dataset/test_dataset.txt")
    clf = FLRegressor(n_trees=10, n_parties=2, mode="horizontal", learning_rate=0.2, max_depth=4, objective="reg:linear")
    clf.fit(x, y)
    y_pred = clf.predict(x)
    rmse = mean_squared_error(y, y_pred, squared=False)
    print("rmse:", rmse)
