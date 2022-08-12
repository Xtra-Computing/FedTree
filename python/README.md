## Install Python Package

We provide a scikit-learn wrapper interface. Before you use the Python interface, you must [install](https://fedtree.readthedocs.io/en/latest/Installation.html) FedTree first. 
Then, you can run the following command to install the Python package from source.
```bash
python setup.py install
```

## Class

We provide two classes, ```FLClassifier``` and ```FLRegressor```, where the first is for classification task and the second is for regression task.

### Parameters
Please refer to [here](https://fedtree.readthedocs.io/en/latest/Parameters.html) for the list of parameters.


### Methods

*fit(X, y)*:\
Fit the FedTree model according to the given training data.

*predict(X)*:\
Perform prediction on samples in X.

*save_model(model_path)*:\
Save the FedTree model to model_path.

*load_model(model_path)*:\
Load the FedTree model from model_path.

## Examples
Users can simply input parameters to these classes, call ```fit()``` and ```predict``` functions like models in scikit-learn.

```bash
from fedtree import FLRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_svmlight_file
x, y = load_svmlight_file("../dataset/test_dataset.txt")
clf = FLRegressor(n_trees=10, n_parties=2, mode="horizontal", learning_rate=0.2, max_depth=4, objective="reg:linear")
clf.fit(x, y)
y_pred = clf.predict(x)
rmse = mean_squared_error(y, y_pred, squared=False)
print("rmse:", rmse)
```

Under ```examples``` directory, you can find three examples on how to use FedTree with Python.
