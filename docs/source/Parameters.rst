APIs/Parameters
===============

We provide two kinds of APIs: command-line interface (CLI) and Python interface. For CLI, users only need to prepare a
configuration file specifying the parameters and call FedTree in a one-line command. For Python interface, users can define
two classes `FLClassifier` and `FLRegressor` with the parameters and use them in a scikit-learn style (see `here <https://github.com/Xtra-Computing/FedTree/tree/main/python>`__).
The parameters are below.

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

* ``partition`` [default = ``0``, type = bool]
    - ``0``: each party has a prepared local dataset
    - ``1``: there is a global dataset and users require FedTree to partition it to multiple subsets to simulate federated setting.

* ``partition_mode`` [default=``horizontal``, type=string]
    - ``horizontal``: horizontal data partitioning
    - ``vertical``: vertical data partitioning

* ``ip_address`` [default=``localhost``, type=string, alias: ``server_ip_address``]
    - The ip address of the server in distributed FedTree.

* ``data_format`` [default=``libsvm``, type=string]
    - ``libsvm``: the input data is in a libsvm format (label feature_id1:feature_value1  feature_id2:feature_value2). See `here <https://github.com/Xtra-Computing/FedTree/blob/main/dataset/test_dataset.txt>`__ for an example.
    - ``csv``: the input data is in a csv format (the first row is the header and the other rows are feature values). See `here <https://github.com/Xtra-Computing/FedTree/blob/main/dataset/credit/credit_vertical_p0_withlabel.csv>`__ for an example.

* ``n_features`` [default=-1, type=int]
    - Number of features of the datasets. It needs to be specified when conducting horizontal FedTree with sparse datasets.

* ``propose_split`` [default=``server``, type=string]
    - ``server``: the server proposes candidate split points according to the range of each feature in horizontal FedTree.
    - ``party``: the parties propose possible split points. Then, the server merge them and sample at most num_max_bin candidate split points in horizontal FedTree.

Parameters for GBDTs
--------------------

* ``data`` [default=``../dataset/test_dataset.txt``, type=string, alias: ``path``]
    - The path to the training dataset(s). In simulation, if multiple datasets need to be loaded where each dataset represents a party, specify the paths seperated with comma.

* ``model_path`` [default=``fedtree.model``, type=string]
    - The path to save/load the model.

* ``verbose`` [default=1, type=int]
    - Printing information: 0 for silence, 1 for key information and 2 for more information.

* ``depth`` [default=6, type=int]

    - The maximum depth of the decision trees. Shallow trees tend to have better generality, and deep trees are more likely to overfit the training data.

* ``n_trees`` [default=40, type=int]

    - The number of training iterations. ``n_trees`` equals to the number of trees in GBDTs.


* ``max_num_bin`` [default=32, type=int]

    - The maximum number of bins in a histogram. The value needs to be smaller than 256.

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

* ``privacy_method`` [default = ``none``, type=string]
    - ``none``: no additional method is used to protect the communicated messages (raw data is not transferred).
    - ``he``: use homomorphic encryption to protect the communicated messages (for vertical FedTree).
    - ``sa``: use secure aggregation to protect the communicated messages (for horizontal FedTree).
    - ``dp``: use differential privacy to protect the communicated messages (currently only works for vertical FL with single machine simulation).


* ``privacy_budget`` [default=10, type=float]
    - Total privacy budget if using differential privacy.
