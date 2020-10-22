//
// Created by kellyyung on 19/10/2020.
//

#include <conio.h>
#include <cstdio>
#include "HomomorphicEncrpytion.h"
#include "Python.h"

//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_HE_H
#define FEDTREE_HE_H

// Todo: addtively homomorphic encryption: generate key pair, encryption, decrption


class HomomorphicEncryption {
public:

    HomomorphicEncryption() {

    }
};

int main()
{
    PyObject* pInt;

    Py_Initialize();

    PyRun_SimpleString("print('Hello World from Embedded Python!!!')");

    Py_Finalize();

    printf("\nPress any key to exit...\n");
    if(!_getch()) _getch();
    return 0;
}
#endif //FEDTREE_HE_H
