Installation
============

Here is the guide for the installation of FedTree.



**Contents**

-  `Prerequisites <#prerequisites>`__

-  `Install Fedtree <#install-fedtree>`__

-  `Build on Linux <#build-on-linux>`__

-  `Build on MacOS <#build-on-macos>`__

Prerequisites
~~~~~~~~~~~~~

* `CMake <https://cmake.org/>`_ 3.15 or above
* `GMP <https://gmplib.org/>`_ library
* `NTL <https://libntl.org/>`_ library

You can follow the following commands to install NTL library.

    .. code::

        wget https://libntl.org/ntl-11.4.4.tar.gz
        tar -xvf ntl-11.4.4.tar.gz
        cd ntl-11.4.4/src
        ./configure
        make
        make check
        sudo make install

If you install the NTL library at another location, please pass the location to the `NTL_PATH` when building the library (e.g., `cmake .. -DNTL_PATH="PATH_TO_NTL"`).

Clone Install submodules
~~~~~~~~~~~~~~~~~~~~~~~~

Run the following commands:

    .. code::

        git clone https://github.com/Xtra-Computing/FedTree
        git submodule init
        git submodule update

Build on Linux
~~~~~~~~~~~~~~
Run the following commands:

    .. code::

        # under the directory of FedTree
        mkdir build && cd build
        cmake ..
        make -j

Build on MacOS
~~~~~~~~~~~~~~
On MacOS, you can use Apple Clang to build FedTree.

Build with Apple Clang
^^^^^^^^^^^^^^^^^^^^^^
Install `libomp` if you haven't:

    .. code::

        brew install libomp

Run the following commands:

    .. code::

        mkdir build
        cd build
        cmake -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
          -DOpenMP_C_LIB_NAMES=omp \
          -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
          -DOpenMP_CXX_LIB_NAMES=omp \
          -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib \
          ..
        make -j

Building Options
~~~~~~~~~~~~~~~~
There are the following building options passing with cmake.

* ``USE_CUDA`` [default = ``OFF``]: Whether using GPU to accelerate homomorphic encryption or not.

* ``DISTRIBUTED`` [default = ``ON``]: Whether building distributed version of FedTree or not.

* ``NTL_PATH`` [default = ``/usr/local``]: The PATH to the NTL library.

For example, if you want to build a version with GPU acceleration, distributed version with NTL library under /home/NTL directory, you can use the following command.

    .. code::

        cmake .. -DUSE_CUDA=ON -DDISTRIBUTED=ON -DNTL_PATH="/home/NTL"
        make -j


