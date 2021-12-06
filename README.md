[Documentation](https://fedtree.readthedocs.io/en/latest/index.html)

# Overview
**FedTree** is a federated learning system for tree-based models. It is designed to be highly **efficient**, **effective**,
and **secure**. It is **under development** and has the following features currently.

- Federated training of gradient boosting decision trees.
- Parallel computing on multi-core CPUs.
- Supporting homomorphic encryption and differential privacy.
- Supporting classification and regression.

The overall architecture of FedTree is shown below.
![FedTree_archi](./docs/source/images/fedtree_archi.png)

# Getting Started
You can refer to our primary documentation [here](https://fedtree.readthedocs.io/en/latest/index.html).
## Prerequisites
* [CMake](https://cmake.org/) 3.15 or above
* [NTL](https://libntl.org/) library

You can follow the following commands to install NTL library.

```
wget https://libntl.org/ntl-11.4.4.tar.gz
tar -xvf ntl-11.4.4.tar.gz
cd ntl-11.4.4/src
./configure SHARED=on
make
make check
sudo make install
```


If you install the NTL library at another location, please also modify the CMakeList files of FedTree accordingly (line 64 of CMakeLists.txt).
## Install submodules
```
git submodule init src/test/googletest
git submodule init thrust
git submodule update
```

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

# Features in development
The following features are in development.

- Distributed Computing.
- Training on GPUs.
- Federated Training of Random Forests.
- Python interfaces.

# Other information
FedTree is built based on [ThunderGBM](https://github.com/Xtra-Computing/thundergbm), which is a fast GBDTs and Radom Forests training system on GPUs.
