Experiments
===========
Here we present some preliminary experimental results. We use two UCI datasets, `adult <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a>`__ and `abalone <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#abalone>`_ for experiments.
The adult dataset is a classification dataset and the abalone is a regression dataset. We use FedTree-Hori to denote the horizontal FedTree and FedTree-Verti to denote the vertical FedTree.

Baselines: Homo-SecureBoost and Hetero-SecureBoost. Both approaches are from `FATE <https://github.com/FederatedAI/FATE>`_.


Standalone Simulation
~~~~~~~~~~~~~~~~~~~~~
For the standalone simulation, we use a machine with 64*Intel Xeon Gold 6226R CPUs and 8*NVIDIA GeForce RTX 3090 to conduct experiments.
We allocate each experiment with 16 threads. By default, we set the number of parties to 2, the number of trees to 50, learning rate to 0.1, the maximum depth of tree to 6, and the maximum number of bins to 255.
The other parameters of all approaches are set to the default setting of FedTree.

Effectiveness
^^^^^^^^^^^^^
We first compare the accuracy of federated training and centralized training using `XGBoost <https://github.com/dmlc/xgboost>`_ and `ThunderGBM <https://github.com/Xtra-Computing/thundergbm>`_. The results are shown below.
We report AUC for adult and RMSE for abalone. We can observe that the performance of FedTree is same as ThunderGBM. Also, SA and HE do not affect the model performance.

+----------+---------+------------+--------------+-----------------+---------------+------------------+------------------+--------------------+
| datasets | XGBoost | ThunderGBM | FedTree-Hori | FedTree-Hori+SA | FedTree-Verti | FedTree-Verti+HE | Homo-SecureBoost | Hetero-SecureBoost |
+----------+---------+------------+--------------+-----------------+---------------+------------------+------------------+--------------------+
|    a9a   |  0.914  |    0.914   |     0.914    |      0.914      |     0.914     |       0.914      |       0.912      |        0.914       |
+----------+---------+------------+--------------+-----------------+---------------+------------------+------------------+--------------------+
|  abalone |   1.53  |    1.57    |     1.57     |       1.57      |      1.56     |       1.57       |       1.56       |        0.001       |
+----------+---------+------------+--------------+-----------------+---------------+------------------+------------------+--------------------+

Efficiency
^^^^^^^^^^

We compare the efficiency of FedTree-Hori with Homo-SecureBoost of FATE. The results are shown below. We present the trainig time (s) per tree.
Note that FedTree-Hori+SA achieves the same security guarantee as Homo-SecureBoost. The speedup is the computed by the improvement of FedTree-Hori+SA over Homo-SecureBoost, which is quite significant.



+----------+--------------+-----------------+------------------+---------+
| datasets | FedTree-Hori | FedTree-Hori+SA | Homo-SecureBoost | Speedup |
+----------+--------------+-----------------+------------------+---------+
|    a9a   |     0.09     |      0.098      |       8.76       |   89.4  |
+----------+--------------+-----------------+------------------+---------+
|  abalone |     0.11     |       0.19      |        7.7       |   40.5  |
+----------+--------------+-----------------+------------------+---------+


We compare the efficiency of FedTree-Verti with Hetero-SecureBoost of FATE.
We present the trainig time (s) per tree. Note that FedTree-Verti+HE achieves the same security guarantee as SecureBoost.
The speedup is the improvement of FedTree-Verti + HE (CPU) over FATE. FedTree is still much faster than SecureBoost. Moreover, FedTree can utilize GPU to accelerate the HE computation.

+----------+---------------+------------------------+------------------------+--------------------+---------+
| datasets | FedTree-Verti | FedTree-Verti+HE (CPU) | FedTree-Verti+HE (GPU) | Hetero-SecureBoost | Speedup |
+----------+---------------+------------------------+------------------------+--------------------+---------+
|    a9a   |      0.11     |          5.25          |          3.24          |        34.02       |   6.48  |
+----------+---------------+------------------------+------------------------+--------------------+---------+
|  abalone |      0.05     |          7.43          |           6.5          |        15.7        |   2.11  |
+----------+---------------+------------------------+------------------------+--------------------+---------+


Distributed Computing
~~~~~~~~~~~~~~~~~~~~~
For distributed setting, we use a cluster with 5 machines, where each machine has two Intel Xeon E5-2680 14 core CPUs.
We set the number of parties to 4, where each party hosts a machine. The results are shown below. Here Homo-SecureBoost (from FATE) and FedTree-Hori+SA have the same security level.
We can observe that both horizontal and vertical FedTree are faster than FATE.

+----------+------------------+-------------------+---------+-------------+------------------+---------+
| datasets | Homo-SecureBoost | FedTree-Hori + SA | Speedup | SecureBoost | FedTree-Verti+HE | Speedup |
+----------+------------------+-------------------+---------+-------------+------------------+---------+
|    a9a   |       214.7      |       124.4       |   1.7   |    505.4    |       93.2       |   5.4   |
+----------+------------------+-------------------+---------+-------------+------------------+---------+
|  abalone |       256.3      |       156.8       |   1.6   |    299.8    |       143.5      |   2.1   |
+----------+------------------+-------------------+---------+-------------+------------------+---------+