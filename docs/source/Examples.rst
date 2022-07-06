Examples
========

Here we present several additional examples of using FedTree.

Distributed Horizontal FedTree with Secure Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the horizontal FedTree, the parties have their local datasets with the same feature space but different sample spaces.
Also, in each machine, a configuration file needs to be prepared.
We take UCI `Adult <https://archive.ics.uci.edu/ml/datasets/adult>`_ dataset as an example (partitioned data provided in `here <https://github.com/Xtra-Computing/FedTree/tree/main/dataset/adult>`__).

In the server machine, the configuration file `server.conf` can be:

    .. code::

        test_data=./dataset/adult/a9a_horizontal_test
        n_parties=2
        objective=binary:logistic
        mode=horizontal
        partition=0
        privacy_tech=sa
        learning_rate=0.1
        max_depth=6
        n_trees=50

In the above configuration file, it needs to specifies number of parties, objective function, mode, privacy techniques, and other parameters for the GBDT model.
The `test_data` specifies the dataset for testing.

Supposing the ip address of the server is a.b.c.d, in the party machine 1, the configuration file `party1.conf` can be:

    .. code::

        data=./dataset/adult/a9a_horizontal_p0
        test_data=./dataset/adult/a9a_horizontal_test
        model_path=p1.model
        n_parties=2
        objective=binary:logistic
        mode=horizontal
        partition=0
        privacy_tech=sa
        learning_rate=0.1
        max_depth=6
        n_trees=50
        ip_address=a.b.c.d

The difference between `party1.conf` and `server.conf` is that `party1.conf` needs to specify the path to the local data and the ip address of the server.
Similarly, we can have a configuration file for each party machine by changing the `data` (and `model_path` if needed). Then, we can run the following commands in the corresponding machines.

    .. code::

        # under 'FedTree' directory
        # under server machine
        ./build/bin/FedTree-distributed-server ./server.conf
        # under party machine 1
        ./build/bin/FedTree-distributed-party ./party1.conf 0
        # under party machine 2
        ./build/bin/FedTree-distributed-party ./party2.conf 1
        ......

In the above commands, the party machines need to add an additional input ID starting from 0 as its party ID.

Distributed Vertical FedTree with Homomorphic Encryption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the vertical FedTree, the parties have their local datasets with the same sample space but different feature spaces.
Moreover, at least one party has the labels of the samples. We need to specify one of the parties that has labels as the aggregator.
Suppose party machine 1 is the aggregator. Then, we need to write a server configuration file `server.conf`, e.g.,

    .. code::

        data=./dataset/adult/a9a_vertical_p0
        test_data=./dataset/adult/a9a_vertical_test
        n_parties=2
        mode=vertical
        partition=0
        reorder_label=1
        objective=binary:logistic
        privacy_tech=he
        learning_rate=0.1
        max_depth=6
        n_trees=50

For each party machine, supposing the ip address of the aggregator is a.b.c.d, we need to write a configuration file, e.g., `party1.conf` in party 1

    .. code::

        data=./dataset/adult/a9a_vertical_p0
        test_data=./dataset/adult/a9a_vertical_test
        model_path=p1.model
        n_parties=2
        mode=vertical
        partition=0
        reorder_label=1
        objective=binary:logistic
        privacy_tech=he
        learning_rate=0.1
        max_depth=6
        n_trees=50
        ip_address=a.b.c.d

Then, we can run the following commands in the corresponding machines:

    .. code::

        #under aggregator machine (i.e., party machine 1)
        ./build/bin/FedTree-distributed-server ./server.conf
        #under party machine 1
        ./build/bin/FedTree-distributed-party ./party1.conf 0
        #under party machine 2
        ./build/bin/FedTree-distributed-party ./party2.conf 1







