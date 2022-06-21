from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error, accuracy_score

import numpy as np
import scipy.sparse as sp
import statistics

from sklearn.utils import check_X_y

from ctypes import *
from os import path
from sys import platform

from ctypes import *
from os import path
from sys import platform

dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
    shared_library_name = "libFedTree.so"
elif platform == "win32":
    shared_library_name = "libFedTree.dll"
elif platform == "darwin":
    shared_library_name = "libFedTree.dylib"
else:
    raise EnvironmentError("OS not supported!")

if path.exists(path.abspath(path.join(dirname, shared_library_name))):
    lib_path = path.abspath(path.join(dirname, shared_library_name))
else:
    lib_path = path.join(dirname, "../../build/lib", shared_library_name)

if path.exists(lib_path):
    fedtree = CDLL(lib_path)
else:
    raise RuntimeError("Please build the library first!")

OBJECTIVE_TYPE = ['reg:linear', 'reg:logistic', 'binary:logistic',
                  'multi:softprob', 'multi:softmax', 'rank:pairwise', 'rank:ndcg']

ESTIMATOR_TYPE = ['classifier', 'regressor']

fedtreeBase = BaseEstimator
fedtreeRegressorBase = RegressorMixin
fedtreeClassifierBase = ClassifierMixin

class FLModel(fedtreeBase):

    def __init__(self, n_parties, partition, alpha, n_hori, n_verti, mode, partition_mode, privacy_tech, propose_split,
                 merge_histogram, variance, privacy_budget, max_depth, n_trees, min_child_weight, lambda_ft, gamma,
                 column_sampling_rate, verbose, n_parallel_trees, learning_rate, objective, num_class, n_device, max_num_bin,
                 seed, ins_bagging_fraction, reorder_label, bagging, constant_h, tree_method):
        # Federated Learning related variables
        self.n_parties = n_parties
        self.partition = partition
        self.alpha = alpha
        self.n_hori = n_hori
        self.n_verti = n_verti
        self.mode = mode
        self.partition_mode = partition_mode
        self.privacy_tech = privacy_tech
        self.propose_split = propose_split
        self.merge_histogram = merge_histogram
        self.variance = variance
        self.privacy_budget = privacy_budget
        self.seed = seed
        self.ins_bagging_fraction = ins_bagging_fraction
        # GBDT related variables
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.min_child_weight = min_child_weight
        self.lambda_ft = lambda_ft
        self.gamma = gamma
        self.column_sampling_rate = column_sampling_rate
        self.verbose = verbose
        self.n_parallel_trees = n_parallel_trees
        self.learning_rate = learning_rate
        self.objective = objective
        self.num_class = num_class
        self.n_device = n_device
        self.max_num_bin = max_num_bin
        self.tree_method = tree_method
        self.path = path
        self.model = None
        self.tree_per_iter = -1
        self.group_label = None
        self.bagging = bagging
        self.reorder_label = reorder_label
        self.constant_h = constant_h

    def fit(self, X, y, groups=None):
        if self.model is not None:
            fedtree.model_free(byref(self.model))
            self.model = None
        sparse = sp.issparse(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')

        fit = self._sparse_fit

        fit(X, y, groups=groups)
        return self

    def _construct_groups(self, groups):
        in_groups = None
        num_groups = 0
        if groups is not None:
            num_groups = len(groups)
            groups = np.asarray(groups, dtype=np.int32, order='C')
            in_groups = groups.ctypes.data_as(POINTER(c_int32))
        return in_groups, num_groups

    def _sparse_fit(self, X, y, groups=None):
        X.data = np.asarray(X.data, dtype=np.float32, order='C')
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(c_float))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        y = np.asarray(y, dtype=np.float32, order='C')
        label = y.ctypes.data_as(POINTER(c_float))
        in_groups, num_groups = self._construct_groups(groups)
        group_label = (c_float * len(set(y)))()
        n_class = (c_int * 1)()
        n_class[0] = self.num_class
        tree_per_iter_ptr = (c_int * 1)()
        self.model = (c_long * 1)()
        fedtree.fit(self.n_parties, self.partition, c_float(self.alpha), self.n_hori, self.n_verti, self.mode.encode('utf-8'),
                    self.partition_mode.encode('utf-8'), self.privacy_tech.encode('utf-8'), self.propose_split.encode('utf-8'), self.merge_histogram.encode('utf-8'), c_float(self.variance), c_float(self.privacy_budget),
                    self.max_depth, self.n_trees, c_float(self.min_child_weight), c_float(self.lambda_ft), c_float(self.gamma), c_float(self.column_sampling_rate),
                    self.verbose, self.bagging, self.n_parallel_trees, c_float(self.learning_rate), self.objective.encode('utf-8'), n_class, self.n_device, self.max_num_bin,
                    self.seed, c_float(self.ins_bagging_fraction), self.reorder_label, c_float(self.constant_h),
                    X.shape[0], data, indptr, indices, label, self.tree_method, byref(self.model), tree_per_iter_ptr, group_label,
                    in_groups, num_groups)
        self.num_class = n_class[0]
        self.tree_per_iter = tree_per_iter_ptr[0]
        self.group_label = [group_label[idx] for idx in range(len(set(y)))]
        if self.model is None:
            print("The model returned is empty!")
            exit()

    def predict(self, X, groups=None):
        if self.model is None:
            print("Please train the model first or load model from file!")
            raise ValueError
        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X.data = np.asarray(X.data, dtype=np.float32, order='C')
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(c_float))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        if(self.objective != 'multi:softprob'):
            self.predict_label_ptr = (c_float * X.shape[0])()
        else:
            temp_size = X.shape[0] * self.num_class
            self.predict_label_ptr = (c_float * temp_size)()
        if self.group_label is not None:
            group_label = (c_float * len(self.group_label))()
            group_label[:] = self.group_label
        else:
            group_label = None
        in_groups, num_groups = self._construct_groups(groups)
        fedtree.predict(
            X.shape[0],
            data,
            indptr,
            indices,
            self.predict_label_ptr,
            byref(self.model),
            self.n_trees,
            self.tree_per_iter,
            self.objective.encode('utf-8'),
            self.num_class,
            c_float(self.learning_rate),
            group_label,
            in_groups, num_groups, self.verbose, self.bagging
        )
        predict_label = [self.predict_label_ptr[index] for index in range(0, X.shape[0])]
        self.predict_label = np.asarray(predict_label)
        return self.predict_label

    def predict_proba(self, X, groups=None):
        if self.model is None:
            print("Please train the model first or load model from file!")
            raise ValueError
        if not ("binary" in self.objective or "multi" in self.objective):
            print("Only classification supports predict_proba!")
            raise ValueError
        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X.data = np.asarray(X.data, dtype=np.float32, order='C')
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(c_float))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        if("multi" not in self.objective):
            self.predict_raw = (c_float * X.shape[0])()
        else:
            temp_size = X.shape[0] * self.num_class
            self.predict_raw = (c_float * temp_size)()
        if self.group_label is not None:
            group_label = (c_float * len(self.group_label))()
            group_label[:] = self.group_label
        else:
            group_label = None
        in_groups, num_groups = self._construct_groups(groups)
        fedtree.predict_proba(
            X.shape[0],
            data,
            indptr,
            indices,
            self.predict_raw,
            byref(self.model),
            self.n_trees,
            self.tree_per_iter,
            self.objective.encode('utf-8'),
            self.num_class,
            c_float(self.learning_rate),
            group_label,
            in_groups, num_groups, self.verbose, self.bagging
        )

        if "binary" in self.objective:
            predict_raw = np.asarray([self.predict_raw[index] for index in range(0, X.shape[0])])
            predict_proba = 1/(1+np.exp(-predict_raw))
            self.predict_proba = np.c_[1-predict_proba, predict_proba]
        elif self.objective == "multi:softmax":
            predict_raw = np.asarray([self.predict_raw[index] for index in range(0, X.shape[0] * self.num_class)])
            self.predict_proba = predict_raw.reshape(X.shape[0], self.num_class)
        return self.predict_proba



    def save_model(self, model_path):
        if self.model is None:
            print("Please train the model first or load model from file!")
            raise ValueError
        if self.group_label is not None:
            group_label = (c_float * len(self.group_label))()
            group_label[:] = self.group_label
        fedtree.save_model(
            model_path.encode('utf-8'),
            self.objective.encode('utf-8'),
            c_float(self.learning_rate),
            self.num_class,
            self.n_trees,
            self.tree_per_iter,
            byref(self.model),
            group_label
        )

    def load_model(self, model_path):
        self.model = (c_long * 1)()
        learning_rate = (c_float * 1)()
        n_class = (c_int * 1)()
        n_trees = (c_int * 1)()
        tree_per_iter = (c_int * 1)()
        fedtree.load_model(
            model_path.encode('utf-8'),
            learning_rate,
            n_class,
            n_trees,
            tree_per_iter,
            self.objective.encode('utf-8'),
            byref(self.model)
        )
        if self.model is None:
            raise ValueError("Model is None.")
        self.learning_rate = learning_rate[0]
        self.num_class = n_class[0]
        self.n_trees = n_trees[0]
        self.tree_per_iter = tree_per_iter[0]
        self.group_label = None
        # group_label = (c_float * self.num_class)()
        # thundergbm.load_config(
        #     model_path.encode('utf-8'),
        #     group_label
        # )
        # self.group_label = [group_label[idx] for idx in range(self.num_class)]


    def __del__(self):
        if self.model is not None:
            fedtree.model_free(byref(self.model))

    # Cross Validation
    def cv(self, X, y, folds=None, nfold=5, shuffle=True, seed=0):
        if self._impl == 'ranker':
            print("Cross-validation for ## Ranker ## have not been supported yep..")
            return

        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        y = np.asarray(y, dtype=np.float32, order='C')
        n_instances = X.shape[0]
        if folds is not None:
            #use specified validation set
            train_idset = [x[0] for x in folds]
            test_idset = [x[1] for x in folds]
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(n_instances)
            else:
                randidx = np.arange(n_instances)
            kstep = int(n_instances / nfold)
            test_idset = [randidx[i: i + kstep] for i in range(0, n_instances, kstep)]
            train_idset = [np.concatenate([test_idset[i] for i in range(nfold) if k != i]) for k in range(nfold)]
        # to be optimized: get score in fit; early stopping; more metrics;
        train_score_list = []
        test_score_list = []
        for k in range(nfold):
            X_train = X[train_idset[k],:]
            X_test = X[test_idset[k],:]
            y_train = y[train_idset[k]]
            y_test = y[test_idset[k]]
            self.fit(X_train, y_train)
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            if self._impl == 'classifier':
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test,y_test_pred)
                train_score_list.append(train_score)
                test_score_list.append(test_score)
            elif self._impl == 'regressor':
                train_score = mean_squared_error(y_train,y_train_pred)
                test_score = mean_squared_error(y_test, y_test_pred)
                train_score_list.append(train_score)
                test_score_list.append(test_score)
        self.eval_res = {}
        if self._impl == 'classifier':
            self.eval_res['train-accuracy-mean']= statistics.mean(train_score_list)
            self.eval_res['train-accuracy-std']= statistics.stdev(train_score_list)
            self.eval_res['test-accuracy-mean'] = statistics.mean(test_score_list)
            self.eval_res['test-accuracy-std'] = statistics.stdev(test_score_list)
            print("mean train accuracy:%.6f+%.6f" %(statistics.mean(train_score_list), statistics.stdev(train_score_list)))
            print("mean test accuracy:%.6f+%.6f" %(statistics.mean(test_score_list), statistics.stdev(test_score_list)))
        elif self._impl == 'regressor':
            self.eval_res['train-RMSE-mean']= statistics.mean(train_score_list)
            self.eval_res['train-RMSE-std']= statistics.stdev(train_score_list)
            self.eval_res['test-RMSE-mean'] = statistics.mean(test_score_list)
            self.eval_res['test-RMSE-std'] = statistics.stdev(test_score_list)
            print("mean train RMSE:%.6f+%.6f" %(statistics.mean(train_score_list), statistics.stdev(train_score_list)))
            print("mean test RMSE:%.6f+%.6f" %(statistics.mean(test_score_list), statistics.stdev(test_score_list)))
        return self.eval_res

class FLClassifier(FLModel, fedtreeClassifierBase):
    _impl = 'classifier'

    def __init__(self, n_parties=2, partition=1, alpha=100, n_hori=2, n_verti=2, mode="horizontal",
                 partition_mode="horizontal", privacy_tech="none", propose_split="server", merge_histogram="server",
                 variance=200, privacy_budget=10, max_depth=6, n_trees=40, min_child_weight=1, lambda_ft=1,
                 gamma=1, column_sampling_rate=1, verbose=1, n_parallel_trees=1, learning_rate=1, objective="binary:logistic",
                 num_class=1, n_device=1, max_num_bin=255, seed=36, ins_bagging_fraction=1.0, reorder_label=0, bagging = 0,
                 constant_h = 0.0, tree_method="auto"):
        super().__init__(n_parties=n_parties, partition=partition, alpha=alpha, n_hori=n_hori, n_verti=n_verti,
                         mode=mode, partition_mode=partition_mode, privacy_tech=privacy_tech, propose_split=propose_split,
                         merge_histogram=merge_histogram, variance=variance, privacy_budget=privacy_budget, max_depth=max_depth,
                         n_trees=n_trees, min_child_weight=min_child_weight, lambda_ft=lambda_ft, gamma=gamma,
                         column_sampling_rate=column_sampling_rate, verbose=verbose, n_parallel_trees=n_parallel_trees,
                         learning_rate=learning_rate, objective=objective, num_class=num_class, n_device=n_device,
                         max_num_bin=max_num_bin, seed=seed, ins_bagging_fraction=ins_bagging_fraction,
                         reorder_label=reorder_label, bagging=bagging, constant_h=constant_h, tree_method=tree_method)

class FLRegressor(FLModel, fedtreeRegressorBase):
    _impl = 'regressor'
    def __init__(self, n_parties=2, partition=1, alpha=100, n_hori=2, n_verti=2, mode="horizontal",
                 partition_mode="horizontal", privacy_tech="none", propose_split="server", merge_histogram="server",
                 variance=200, privacy_budget=10, max_depth=6, n_trees=40, min_child_weight=1, lambda_ft=1,
                 gamma=1, column_sampling_rate=1, verbose=1, n_parallel_trees=1, learning_rate=1, objective="reg:linear",
                 num_class=1, n_device=1, max_num_bin=255, seed=36, ins_bagging_fraction=1.0, reorder_label=0, bagging = 0,
                 constant_h = 0.0, tree_method="auto"):
        super().__init__(n_parties=n_parties, partition=partition, alpha=alpha, n_hori=n_hori, n_verti=n_verti,
                         mode=mode, partition_mode=partition_mode, privacy_tech=privacy_tech, propose_split=propose_split,
                         merge_histogram=merge_histogram, variance=variance, privacy_budget=privacy_budget, max_depth=max_depth,
                         n_trees=n_trees, min_child_weight=min_child_weight, lambda_ft=lambda_ft, gamma=gamma,
                         column_sampling_rate=column_sampling_rate, verbose=verbose, n_parallel_trees=n_parallel_trees,
                         learning_rate=learning_rate, objective=objective, num_class=num_class, n_device=n_device,
                         max_num_bin=max_num_bin, seed=seed, ins_bagging_fraction=ins_bagging_fraction,
                         reorder_label=reorder_label, bagging=bagging, constant_h=constant_h, tree_method=tree_method)
