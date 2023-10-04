from fedtree import FLModel, fedtree, fedtreeClassifierBase, fedtreeRegressorBase
from ctypes import *
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_X_y


class DistributedFLModel(FLModel):
    def __init__(
            self, pid, server_addr, server_port, n_parties, partition, alpha, n_hori,
            n_verti, mode, partition_mode, privacy_tech, propose_split, merge_histogram,
            variance, privacy_budget, max_depth, n_trees, min_child_weight, lambda_ft,
            gamma, column_sampling_rate, verbose, n_parallel_trees, learning_rate,
            objective, num_class, n_device, max_num_bin, seed, ins_bagging_fraction,
            reorder_label, bagging, constant_h, tree_method, use_double):
        super().__init__(n_parties, partition, alpha, n_hori, n_verti, mode,
                         partition_mode, privacy_tech, propose_split, merge_histogram, variance,
                         privacy_budget, max_depth, n_trees, min_child_weight, lambda_ft, gamma,
                         column_sampling_rate, verbose, n_parallel_trees, learning_rate, objective,
                         num_class, n_device, max_num_bin, seed, ins_bagging_fraction, reorder_label,
                         bagging, constant_h, tree_method, use_double,)

        try:
            getattr(fedtree, "fit_distributed")
        except AttributeError:
            print(
                "The library is not built for distributed settings, please build it with -DDISTRIBUTED=ON.")
            exit(1)

        self.pid = pid
        self.server_addr = server_addr
        self.server_port = server_port

    def fit(self, X, y, groups=None):
        if self.model is not None:
            fedtree.model_free(byref(self.model))
            self.model = None
        sparse = sp.issparse(X)
        if sparse is False:
            # potential bug: csr_matrix ignores all zero values in X
            X = sp.csr_matrix(X)
        X, y = check_X_y(X, y, dtype=np.float64,
                         order="C", accept_sparse="csr")

        fit = self._sparse_fit

        fit(X, y, groups=groups)
        return self

    def _sparse_fit(self, X: sp.csr_matrix, y: np.ndarray, groups=None):
        if self.use_double:
            X.data = np.asarray(X.data, dtype=np.float64, order="C")
        else:
            X.data = np.asarray(X.data, dtype=np.float32, order="C")
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(self.float_type))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        if self.use_double:
            y = np.asarray(y, dtype=np.float64, order="C")
        else:
            y = np.asarray(y, dtype=np.float32, order="C")
        label = y.ctypes.data_as(POINTER(self.float_type))
        in_groups, num_groups = self._construct_groups(groups)
        group_label = (self.float_type * len(set(y)))()
        n_class = (c_int * 1)()
        n_class[0] = self.num_class
        tree_per_iter_ptr = (c_int * 1)()
        self.model = (c_long * 1)()

        fedtree.fit_distributed(
            self.pid, self.server_addr.encode("utf-8"),
            self.server_port, self.n_parties, self.partition, c_float(self.alpha),
            self.n_hori, self.n_verti, self.mode.encode("utf-8"),
            self.partition_mode.encode("utf-8"),
            self.privacy_tech.encode("utf-8"),
            self.propose_split.encode("utf-8"),
            self.merge_histogram.encode("utf-8"),
            c_float(self.variance),
            c_float(self.privacy_budget),
            self.max_depth, self.n_trees, c_float(self.min_child_weight),
            c_float(self.lambda_ft),
            c_float(self.gamma),
            c_float(self.column_sampling_rate),
            self.verbose, self.bagging, self.n_parallel_trees,
            c_float(self.learning_rate),
            self.objective.encode("utf-8"),
            n_class, self.n_device, self.max_num_bin, self.seed,
            c_float(self.ins_bagging_fraction),
            self.reorder_label, c_float(self.constant_h),
            X.shape[0],
            data, indptr, indices, label, self.tree_method, byref(self.model),
            tree_per_iter_ptr, group_label, in_groups, num_groups,)

        self.num_class = n_class[0]
        self.tree_per_iter = tree_per_iter_ptr[0]
        self.group_label = [group_label[idx] for idx in range(len(set(y)))]
        if self.model is None:
            print("The model returned is empty!")
            exit()


class DistributedFLClassifier(DistributedFLModel, fedtreeClassifierBase):
    _impl = "classifier"

    def __init__(
            self, pid, server_addr, server_port, n_parties=2, partition=1, alpha=100,
            n_hori=2, n_verti=2, mode="horizontal", partition_mode="horizontal",
            privacy_tech="none", propose_split="server", merge_histogram="server",
            variance=200, privacy_budget=10, max_depth=6, n_trees=40, min_child_weight=1,
            lambda_ft=1, gamma=1, column_sampling_rate=1, verbose=1, n_parallel_trees=1,
            learning_rate=1, objective="binary:logistic", num_class=1, n_device=1,
            max_num_bin=255, seed=36, ins_bagging_fraction=1.0, reorder_label=0,
            bagging=0, constant_h=0.0, tree_method="auto", use_double=0,):
        super().__init__(
            pid=pid, server_addr=server_addr, server_port=server_port,
            n_parties=n_parties, partition=partition, alpha=alpha, n_hori=n_hori,
            n_verti=n_verti, mode=mode, partition_mode=partition_mode,
            privacy_tech=privacy_tech, propose_split=propose_split,
            merge_histogram=merge_histogram, variance=variance,
            privacy_budget=privacy_budget, max_depth=max_depth, n_trees=n_trees,
            min_child_weight=min_child_weight, lambda_ft=lambda_ft, gamma=gamma,
            column_sampling_rate=column_sampling_rate, verbose=verbose,
            n_parallel_trees=n_parallel_trees, learning_rate=learning_rate,
            objective=objective, num_class=num_class, n_device=n_device,
            max_num_bin=max_num_bin, seed=seed, ins_bagging_fraction=ins_bagging_fraction,
            reorder_label=reorder_label, bagging=bagging, constant_h=constant_h,
            tree_method=tree_method, use_double=use_double,)


class DistributedFLRegressor(DistributedFLModel, fedtreeRegressorBase):
    _impl = "regressor"

    def __init__(
            self, pid, server_addr, server_port, n_parties=2, partition=1, alpha=100,
            n_hori=2, n_verti=2, mode="horizontal", partition_mode="horizontal",
            privacy_tech="none", propose_split="server", merge_histogram="server",
            variance=200, privacy_budget=10, max_depth=6, n_trees=40, min_child_weight=1,
            lambda_ft=1, gamma=1, column_sampling_rate=1, verbose=1, n_parallel_trees=1,
            learning_rate=1, objective="reg:linear", num_class=1, n_device=1,
            max_num_bin=255, seed=36, ins_bagging_fraction=1.0, reorder_label=0,
            bagging=0, constant_h=0.0, tree_method="auto", use_double=0,):
        super().__init__(
            pid=pid, server_addr=server_addr, server_port=server_port,
            n_parties=n_parties, partition=partition, alpha=alpha, n_hori=n_hori,
            n_verti=n_verti, mode=mode, partition_mode=partition_mode,
            privacy_tech=privacy_tech, propose_split=propose_split,
            merge_histogram=merge_histogram, variance=variance,
            privacy_budget=privacy_budget, max_depth=max_depth, n_trees=n_trees,
            min_child_weight=min_child_weight, lambda_ft=lambda_ft, gamma=gamma,
            column_sampling_rate=column_sampling_rate, verbose=verbose,
            n_parallel_trees=n_parallel_trees, learning_rate=learning_rate,
            objective=objective, num_class=num_class, n_device=n_device,
            max_num_bin=max_num_bin, seed=seed, ins_bagging_fraction=ins_bagging_fraction,
            reorder_label=reorder_label, bagging=bagging, constant_h=constant_h,
            tree_method=tree_method, use_double=use_double,)
