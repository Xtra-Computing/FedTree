Parameters
==========

Parameters for Federated Setting
--------------------------------

* ``mode`` [default = ``horizontal``, type=string]
    - ``horizontal``: horizontal federated learning
    - ``vertical``: vertical federated learning

* ``num_parties`` [default = ``10``, type = int, alias: ``num_clients``, ``num_devices``]
    - Number of parties

* ``privacy_method`` [default = ``none``, type=string]
    - ``none``: no additional method is used to protect the communicated messages (raw data is not transferred).
    - ``he``: use homomorphic encryption to protect the communicated messages.
    - ``dp``: use differential privacy to protect the communicated messages.

* ``partition_mode`` [default=``iid``, type=string]
    - ``iid``: IID data partitioning
    - ``noniid``: non-IID data partitioning

Parameters for GBDT model
-------------------------

* ``verbose`` [default=1, type=int]
    - Printing information: 0 for silence, 1 for key information and 2 for more information.

* ``depth`` [default=6, type=int]

    - The maximum depth of the decision trees. Shallow trees tend to have better generality, and deep trees are more likely to overfit the training data.

* ``n_trees`` [default=40, type=int]

    - The number of training iterations. ``n_trees`` equals to the number of trees in GBDTs.


* ``max_num_bin`` [default=255, type=int]

    - The maximum number of bins in a histogram.

* ``learning_rate`` [default=1, type=float, alias(only for c++): ``eta``]

    - Valid domain: [0,1]. This option is to set the weight of newly trained tree. Use ``eta < 1`` to mitigate overfitting.

* ``objective`` [default=``reg:linear``, type=string]

    - Valid options include ``reg:linear``, ``reg:logistic``, ``multi:softprob``,  ``multi:softmax``, ``rank:pairwise`` and ``rank:ndcg``.
    - ``reg:linear`` is for regression, ``reg:logistic`` and ``binary:logistic`` are for binary classification.
    - ``multi:softprob`` and ``multi:softmax`` are for multi-class classification. ``multi:softprob`` outputs probability for each class, and ``multi:softmax`` outputs the label only.
    - ``rank:pairwise`` and ``rank:ndcg`` are for ranking problems.

* ``num_class`` [default=1, type=int]
    - Set the number of classes in the multi-class classification. This option is not compulsory.

* ``min_child_weight`` [default=1, type=float]

    - The minimum sum of instance weight (measured by the second order derivative) needed in a child node.

* ``lambda`` [default=1, type=float, alias(only for c++): ``lambda_tgbm`` or ``reg_lambda``]

    - L2 regularization term on weights.

* ``gamma`` [default=1, type=float, alias(only for c++): ``min_split_loss``]

    - The minimum loss reduction required to make a further split on a leaf node of the tree. ``gamma`` is used in the pruning stage.


Parameters for privacy approaches
---------------------------------

* ``privacy_budget`` [default=10, type=float]
    - Total privacy budget if using differential privacy.