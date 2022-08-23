[Documentation](https://fedtree.readthedocs.io/en/latest/index.html)

# Overview
**FedTree** is a federated learning system for tree-based models. It is designed to be highly **efficient**, **effective**,
and **secure**. It has the following features currently.

- Federated training of gradient boosting decision trees.
- Parallel computing on multi-core CPUs and GPUs.
- Supporting homomorphic encryption, secure aggregation and differential privacy.
- Supporting classification and regression.

The overall architecture of FedTree is shown below.
![FedTree_archi](./docs/source/images/fedtree_archi.png)

# Getting Started
You can refer to our primary documentation [here](https://fedtree.readthedocs.io/en/latest/index.html).
## Prerequisites
* [CMake](https://cmake.org/) 3.15 or above
* [GMP](https://gmplib.org/)
* [NTL](https://libntl.org/)
* [gRPC](https://grpc.io/docs/languages/cpp/quickstart/)

You can follow the following commands to install NTL library.

```
wget https://libntl.org/ntl-11.5.1.tar.gz
tar -xvf ntl-11.5.1.tar.gz
cd ntl-11.5.1/src
./configure SHARED=on
make
make check
sudo make install
```


If you install the NTL library at another location, please pass the location to the `NTL_PATH` when building the library (e.g., `cmake .. -DNTL_PATH="PATH_TO_NTL"`).
## Clone and Install submodules
```
git clone https://github.com/Xtra-Computing/FedTree.git
cd FedTree
git submodule init
git submodule update
```
# Standalone Simulation

## Build on Linux

```bash
# under the directory of FedTree
mkdir build && cd build 
cmake ..
make -j
```

## Build on MacOS

### Build with Apple Clang

You need to install ```libomp``` for MacOS.
```
brew install libomp
```

Install FedTree:
```bash
# under the directory of FedTree
mkdir build
cd build
cmake -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib \
  ..
make -j
```

## Run training
```bash
# under 'FedTree' directory
./build/bin/FedTree-train ./examples/vertical_example.conf
```


# Distributed Setting
For each machine that participates in FL, it needs to build the library first.
```bash
mkdir build && cd build
cmake .. -DDISTRIBUTED=ON
make -j
```
Then, write your configuration file where you should specify the ip address of the server machine (`ip_address=xxx`). Run `FedTree-distributed-server` in the server machine and run `FedTree-distributed-party` in the party machines. 
Here are two examples for horizontal FedTree and vertical FedTree.

[//]: # (export CPLUS_INCLUDE_PATH=./build/_deps/grpc-src/include/:$CPLUS_INCLUDE_PATH)
[//]: # (export CPLUS_INCLUDE_PATH=./build/_deps/grpc-src/third_party/protobuf/src/:$CPLUS_INCLUDE_PATH)

## Distributed Horizontal FedTree
```bash
# under 'FedTree' directory
# under server machine
./build/bin/FedTree-distributed-server ./examples/adult/a9a_horizontal_server.conf
# under party machine 0
./build/bin/FedTree-distributed-party ./examples/adult/a9a_horizontal_p0.conf 0
# under party machine 1
./build/bin/FedTree-distributed-party ./examples/adult/a9a_horizontal_p1.conf 1
```

## Distributed Vertical FedTree
```bash
# under 'FedTree' directory
# under server (i.e., the party with label) machine 0
./build/bin/FedTree-distributed-server ./examples/credit/credit_vertical_p0_withlabel.conf
# open a new terminal
./build/bin/FedTree-distributed-party ./examples/credit/credit_vertical_p0_withlabel.conf 0
# Under party machine 1
./build/bin/FedTree-distributed-party ./examples/credit/credit_vertical_p1.conf 1
```

# Other information
FedTree is built based on [ThunderGBM](https://github.com/Xtra-Computing/thundergbm), which is a fast GBDTs and Radom Forests training system on GPUs.

# Citation
Please cite our paper if you use FedTree in your work.
```
@misc{fedtree,
  title = {FedTree: A Fast, Effective, and Secure Tree-based Federated Learning System},
  author={Li, Qinbin and Cai, Yanzheng and Han, Yuxuan and Yung, Ching Man and Fu, Tianyuan and He, Bingsheng},
  howpublished = {\url{https://github.com/Xtra-Computing/FedTree/blob/main/FedTree_draft_paper.pdf}},
  year={2022}
}
```


