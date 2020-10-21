# FedTree
A tree-based federated learning system.

# Prerequisites
* ```cmake``` 3.15 or above
* ```gcc``` 4.8 or above for Linux and MacOS

# Install submodules
```
git submodule init src/test/googletest
```
For CPUs:
```
git submodule init thrust
```

For GPUs:
```
git submodule init cub
```

Then
```
git submodule update
```


# Build on Linux
With CUDA supports:
```bash
# under the directory of FedTree
mkdir build && cd build && cmake .. && make -j
```
Without CUDA supports (e.g., on CPUs):
```bash
# under the directory of FedTree
mkdir build && cd build 
cmake -DUSE_CUDA=OFF ..
make -j
```

# Build on MacOS

##Build with gcc
Install gcc:
```
brew install gcc
```
Install FedTree without CUDA
```
mkdir build
cd build
cmake -DUSE_CUDA=OFF -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7 .. # replace "7" with version of gcc installed
make -j
```
##Build with Apple Clang
You need to install ```libomp``` for MacOS.
```
brew install libomp
```

Without CUDA supports (e.g., on CPUs):
```bash
# under the directory of FedTree
mkdir build
cd build
cmake -DUSE_CUDA=OFF \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib \
  ..
make -j
```

# Run Tests
```bash
# under 'build' directory
./src/test/FedTree-test
```
