Experiments
===========
Here we present some preliminary experimental results. We use two UCI datasets, `adult <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a>`__ and `abalone <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#abalone>`_ for experiments.
The adult dataset is a classification dataset and the abalone is a regression dataset. We use FedTree-Hori to denote the horizontal FedTree and FedTree-Verti to denote the vertical FedTree.


Standalone Simulation
~~~~~~~~~~~~~~~~~~~~~
For the standalone simulation, we use a machine with 64*Intel Xeon Gold 6226R CPUs and 8*NVIDIA GeForce RTX 3090 to conduct experiments.
We allocate each experiment with 8 threads. By default, we set the number of parties to 2, the number of trees to 40, learning rate to 0.1, and the maximum depth of tree to 6.

Effectiveness
^^^^^^^^^^^^^
We first compare the accuracy of federated training using FedTree and centralized training using `ThunderGBM <https://github.com/Xtra-Computing/thundergbm>`_. The results are shown below.
We report AUC for adult and RMSE for abalone. We can observe that the performance of FedTree is very close to ThunderGBM. Also, SA and HE only change a little on the performance, which is mainly due to the limit of floating point precision.


+----------+------------+--------------+-------------------+---------------+--------------------+
| datasets | ThunderGBM | FedTree-Hori | FedTree-Hori + SA | FedTree-Verti | FedTree-Verti + HE |
+----------+------------+--------------+-------------------+---------------+--------------------+
|   adult  |    0.911   |     0.911    |       0.911       |     0.914     |        0.911       |
+----------+------------+--------------+-------------------+---------------+--------------------+
|  abalone |    1.644   |     1.648    |       1.628       |     1.629     |        1.644       |
+----------+------------+--------------+-------------------+---------------+--------------------+




Efficiency
^^^^^^^^^^

We compare the efficiency of FedTree-Verti with SecureBoost of `FATE <https://github.com/FederatedAI/FATE>`_.
We present the trainig time (s) per tree. The speedup is the improvement of FedTree-Verti + HE (CPU) over FATE.
Note that FedTree-Verti+HE achieves the same security guarantee as SecureBoost. From the table, we can observe that FedTree is much faster than FATE.

+----------+-------------+---------------+--------------------------+--------------------------+---------+
| datasets | SecureBoost | FedTree-Verti | FedTree-Verti + HE (CPU) | FedTree-Verti + HE (GPU) | Speedup |
+----------+-------------+---------------+--------------------------+--------------------------+---------+
|    a9a   |     51.8    |      0.13     |            6.2           |            4.1           |   8.4   |
+----------+-------------+---------------+--------------------------+--------------------------+---------+
|  abalone |     21.4    |      0.13     |            7.5           |            6.2           |   2.9   |
+----------+-------------+---------------+--------------------------+--------------------------+---------+


Distributed Computing
~~~~~~~~~~~~~~~~~~~~~
For distributed setting, we use a cluster with 5 machines, where each machine has 56*Intel Xeon E5-2680 CPUs.
We set the number of parties to 4, where each party hosts a machine. The results are shown below. Here Homo-SecureBoost (from FATE) and FedTree-Hori+SA have the same security level.
We can observe that both horizontal and vertical FedTree are faster than FATE.

+----------+------------------+-------------------+---------+-------------+------------------+---------+
| datasets | Homo-SecureBoost | FedTree-Hori + SA | Speedup | SecureBoost | FedTree-Verti+HE | Speedup |
+----------+------------------+-------------------+---------+-------------+------------------+---------+
|    a9a   |       214.7      |       124.4       |   1.7   |    505.4    |       93.2       |   5.4   |
+----------+------------------+-------------------+---------+-------------+------------------+---------+
|  abalone |       256.3      |       156.8       |   1.6   |    299.8    |       143.5      |   2.1   |
+----------+------------------+-------------------+---------+-------------+------------------+---------+