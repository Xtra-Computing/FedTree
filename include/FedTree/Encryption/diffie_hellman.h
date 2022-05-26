#ifndef FEDTREE_DIFFIE_HELLMAN_H
#define FEDTREE_DIFFIE_HELLMAN_H

#include <NTL/ZZ.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/vector.h>
#include <vector>
#include "FedTree/common.h"
using namespace NTL;
using namespace std;
class DiffieHellman {
public:
    DiffieHellman();

    DiffieHellman& operator=(DiffieHellman source) {
        this->p = source.p;
        this->g = source.g;
//        this->random = source.random;
        return *this;
    }
//    void primegen();

    void init_variables(int n_parties);
    void generate_public_key();
    void compute_shared_keys();
    void generate_noises();
    void decrypt_noises();

    ZZ encrypt(float_type &message, int pid);

    float_type decrypt(ZZ &message, int pid);

    NTL::ZZ p, g;
    ZZ public_key;
    Vec<ZZ> other_public_keys;
    long key_length = 1024;
    int pid;
    int n_parties;


    Vec<ZZ> encrypted_noises;
    Vec<ZZ> received_encrypted_noises;
//private:
    Vec<NTL::ZZ> shared_keys;
    vector<float_type> generated_noises;
    vector<float_type> decrypted_noises;
    unsigned int secret;

};


#endif

