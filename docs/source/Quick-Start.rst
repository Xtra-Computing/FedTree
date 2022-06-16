Quick Start
===========

Here we present an example to simulate vertical federated learning with FedTree to help you understand the procedure of using FedTree.

Prepare a dataset / datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can either prepare a global dataset to simulate the federated setting by partitioning in FedTree or prepare a local dataset for each party.

For the data format, FedTree supports svmlight/libsvm format (each row is an instance with ``label feature_id1:feature_value1  feature_id2:feature_value2 ...``)
and csv format (the first row is the header ``id,label,feature_id1,feature_id2,...`` and the other rows are the corresponding values).
See `here <https://github.com/Xtra-Computing/FedTree/blob/main/dataset/test_dataset.txt>`__ for an example of libsvm format dataset
and `here <https://github.com/Xtra-Computing/FedTree/blob/main/dataset/credit/credit_vertical_p0_withlabel.csv>`__ for an example of csv format dataset.

For classification task, please ensure that the labels of the dataset are organized as ``0 1 2 ...`` (e.g., use labels 0 and 1 for binary classification).

Configure the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~
You can set the parameters in a file, e.g., ``machine.conf`` under ``dataset`` subdirectory.
For example, we can set the following example parameters to run vertical federated learning using homomorphic encryption to protect the communicated message.
For more details about the parameters, please refer to `here <https://fedtree.readthedocs.io/en/latest/Parameters.html#>`__.

    .. code::

        data=./dataset/test_dataset.txt
        test_data=./dataset/test_dataset.txt
        partition_mode=vertical
        n_parties=4
        mode=vertical
        privacy_tech=he
        n_trees=40
        depth=6
        learning_rate=0.2

Run FedTree
~~~~~~~~~~~
After you install FedTree, you can simply run the following commands under ``FedTree`` directory to simulate vertical federated learning in a single machine.

    .. code::

        ./build/bin/FedTree-train ./examples/vertical_example.conf
        ./build/bin/FedTree-predict ./examples/prediction.conf







.. _LibSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/