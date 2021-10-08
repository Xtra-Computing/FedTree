Quick Start
===========

This is an example to run FedTree.

Prepare a dataset
~~~~~~~~~~~~~~~~~
You should first prepare a svmlight/libsvm format dataset. Example datasets can be downloaded from `LIBSVM`_ website.

We have provided ``test_dataset.txt`` under ``dataset`` subdirectory.

Configure the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~
You can set the parameters in a file, e.g., ``machine.conf`` under ``dataset`` subdirectory.
We can set the following example parameters to run vertical federated learning using homomorphic encryption to protect the communicated message.

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
After you install FedTree, you can simply run the following commands under ``FedTree`` directory.

    .. code::

        ./build/bin/FedTree-train ./dataset/vertical_example.conf







.. _LibSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/