Parameters
==========

**Contents**

-  `Parameters for Federated Setting <#parameters-for-federated-setting>`__

-  `Parameters for GBDTs <#parameters-for-gbdts>`__

-  `Parameters for Privacy Protection <#parameters-for-privacy-protection>`__

Parameters for Federated Setting
--------------------------------

* ``mode`` [default = ``horizontal``, type=string]
    - ``horizontal``: horizontal federated learning
    - ``vertical``: vertical federated learning

* ``num_parties`` [default = ``10``, type = int, alias: ``num_clients``, ``num_devices``]
    - Number of parties

* ``privacy_method`` [default = ``none``, type=string]
    - ``none``: no additional method is used to protect the communicated messages (raw data is not transferred).
    - ``he``: use homomorphic encryption to protect the communicated messages (for vertical FedTree).
    - ``sa``: use secure aggregation to protect the communicated messages (for horizontal FedTree).
    - ``dp``: use differential privacy to protect the communicated messages (for simulation).

* ``partition`` or ``simulation`` [default = ``0``, type = bool]
    - ``0``: distributed setting
    - ``1``: standalone simulation

* ``partition_mode`` [default=``iid``, type=string]
    - ``iid``: IID data partitioning
    - ``noniid``: non-IID data partitioning

* ``ip_address`` [default=``localhost``, type=string, alias: ``server_ip_address``]
    - The ip address of the server in distributed FedTree.

* ``data_format`` [default=``libsvm``, type=string]
    - ``libsvm``: the input data is in a libsvm format (label feature_id1:feature_value1  feature_id2:feature_value2). See `here <https://github.com/Xtra-Computing/FedTree/blob/main/dataset/test_dataset.txt>`__ for an example.
    - ``csv``: the input data is in a csv format (the first row is the header and the other rows are feature values). See `here <https://github.com/Xtra-Computing/FedTree/blob/main/dataset/credit/credit_vertical_p0_withlabel.csv>`__ for an example.

* ``reorder_label`` [default=``true``, type=bool]
    - For classification task in standalone simulation, if the labels are not organized as ``0 1 2 ...`` (e.g., the labels are -1 and 1), the users can set `reorder_label` to `true`. For distributed setting, users are suggested to organize the labels in prior and set `reorder_label` to `false`.

Parameters for GBDTs
--------------------

* ``verbose`` [default=1, type=int]
    - Printing information: 0 for silence, 1 for key information and 2 for more information.

* ``depth`` [default=6, type=int]

    - The maximum depth of the decision trees. Shallow trees tend to have better generality, and deep trees are more likely to overfit the training data.

* ``n_trees`` [default=40, type=int]

    - The number of training iterations. ``n_trees`` equals to the number of trees in GBDTs.


* ``max_num_bin`` [default=255, type=int]

    - The maximum number of bins in a histogram.

* ``learning_rate`` [default=1, type=float, alias: ``eta``]

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

* ``lambda`` [default=1, type=float, alias: ``lambda_tgbm`` or ``reg_lambda``]

    - L2 regularization term on weights.

* ``gamma`` [default=1, type=float, alias: ``min_split_loss``]

    - The minimum loss reduction required to make a further split on a leaf node of the tree. ``gamma`` is used in the pruning stage.


Parameters for Privacy Protection
---------------------------------

* ``privacy_budget`` [default=10, type=float]
    - Total privacy budget if using differential privacy.
