from os import path
import setuptools
from shutil import copyfile
from sys import platform

dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
    lib_path = path.abspath(path.join(dirname, '../build/lib/libFedTree.so'))
elif platform == "win32":
    lib_path = path.abspath(path.join(dirname, '../build/bin/Debug/libFedTree.dll'))
elif platform == "darwin":
    lib_path = path.abspath(path.join(dirname, '../build/lib/libFedTree.dylib'))
else:
    print("OS not supported!")
    exit()

copyfile(lib_path, path.join(dirname, "fedtree", path.basename(lib_path)))

setuptools.setup(name="fedtree",
                 version="0.1.0",
                 packages=["fedtree"],
                 package_dir={"python": "fedtree"},
                 package_data={"fedtree": [path.basename(lib_path)]},
                 install_requires=['numpy', 'scipy', 'scikit-learn']
                 )
