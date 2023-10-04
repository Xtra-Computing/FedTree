from os import path
import setuptools
from shutil import copyfile
from sys import platform

dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
    lib_path = path.abspath(
        path.join(dirname, "../build/lib/libFedTree_DIST_SCIKIT.so"))
elif platform == "win32":
    lib_path = path.abspath(
        path.join(dirname, "../build/bin/Debug/libFedTree_DIST_SCIKIT.dll")
    )
elif platform == "darwin":
    lib_path = path.abspath(
        path.join(dirname, "../build/lib/libFedTree_DIST_SCIKIT.dylib"))
else:
    print("OS not supported!")
    exit()

if not path.isfile(lib_path):  # DISTRIBUTED=OFF
    distributed = False
    lib_path = lib_path.replace("_DIST_SCIKIT", "")
    if not path.isfile(lib_path):
        print("Library not found at {}, please build it first".format(lib_path))
        exit()
else:  # DISTRIBUTED=ON
    distributed = True

# rename to libFedTree.xx
lib_dst_path = path.join(dirname, "fedtree", path.basename(
    lib_path).replace("_DIST_SCIKIT", ""))

copyfile(lib_path, lib_dst_path)

setuptools.setup(
    name="fedtree",
    version="1.0.5",
    packages=["fedtree"],
    package_dir={"python": "fedtree"},
    description="A federated learning library for trees",
                 license='Apache-2.0',
                 author='Qinbin Li',
                 author_email='liqinbin1998@gmail.com',
                 url='https://github.com/Xtra-Computing/FedTree',
                 package_data={"fedtree": [lib_dst_path]},
                 install_requires=['numpy', 'scipy', 'scikit-learn'],
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: Apache Software License",
                 ],
                 )
