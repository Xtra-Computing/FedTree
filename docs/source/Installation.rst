Installation
============

Here is the guide for the installation of FedTree.



**Contents**

- `Prerequisites <#prerequisites>`__

-  `Install Fedtree <#install-fedtree>`__

-  `Build on Linux <#build-on-linux>`__

-  `Build on MacOS <#build-on-macos>`__

Prerequisites
~~~~~~~~~~~~~

* `CMake`_ 3.15 or above
* `NTL`_ library

You can follow the following commands to install NTL library.

    .. code::

        wget https://libntl.org/ntl-11.4.4.tar.gz
        tar -xvf ntl-11.4.4.tar.gz
        cd ntl-11.4.4/src
        ./configure
        make
        make check
        sudo make install

If you install the NTL library at another location, please also modify the CMakeList files of FedTree accordingly (line 64 of CMakeLists.txt).

Install FedTree
~~~~~~~~~~~~~~~

Run the following commands:

    .. code::

        git clone https://github.com/Xtra-Computing/FedTree
        git submodule init src/test/googletest
        git submodule init thrust
        git submodule update

Build on Linux
~~~~~~~~~~~~~~
Run the following commands:

    .. code::

        # under the directory of FedTree
        mkdir build && cd build
        cmake -DUSE_CUDA=OFF ..
        make -j

Build on MacOS
~~~~~~~~~~~~~~
On MacOS, you can either use gcc or Apple Clang to build FedTree.

Build with gcc
^^^^^^^^^^^^^^
Install `gcc` if you haven't:

    .. code::

        brew install gcc

Run the following commands:

    .. code::

        mkdir build
        cd build
        cmake -DUSE_CUDA=OFF -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7 .. # replace "7" with version of gcc installed
        make -j

Build with Apple Clang
^^^^^^^^^^^^^^^^^^^^^^
Install `libomp` if you haven't:

    .. code::

        brew install libomp

Run the following commands:

    .. code::

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


.. _CMake: https://cmake.org/
.. _NTL: https://libntl.org/

