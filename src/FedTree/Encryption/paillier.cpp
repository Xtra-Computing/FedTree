#include "FedTree/Encryption/paillier.h"
#include <stdlib.h>

using namespace std;

/* Reference: Paillier, P. (1999, May). Public-key cryptosystems based on composite degree residuosity classes. */


NTL::ZZ Gen_Coprime(const NTL::ZZ &n) {
    /* Coprime generation function. Generates a random coprime number of n.
    *
    * Parameters
    * ==========
    * NTL::ZZ n : a prime number
    *
    * Returns
    * =======
    * NTL:ZZ ret : a random coprime number of n
    */
    NTL::ZZ ret;
    while (true) {
        ret = RandomBnd(n);
        if (NTL::GCD(ret, n) == 1) { return ret; }
    }
}

NTL::ZZ lcm(const NTL::ZZ &x, const NTL::ZZ &y) {
    /* Least common multiple function. Computes the least common multiple of x and y.
     *
     * Parameters
     * ==========
     * NTL::ZZ x, y: signed, arbitrary length integers
     *
     * Returns
     * =======
     * NTL:ZZ lcm : the least common multiple of x and y
     */
    NTL::ZZ lcm;
    lcm = (x * y) / NTL::GCD(x, y);
    return lcm;
}

void GenPrimePair(NTL::ZZ &p, NTL::ZZ &q, long keyLength) {
    /* Prime pair generation function. Generates a prime pair in same bit length.
     *
     * Parameters
     * ==========
     * NTL::ZZ p, q: large primes in same bit length
     * long keyLength: the length of the key
     */
    while (true) {
        long err = 80;
        p = NTL::GenPrime_ZZ(keyLength / 2, err);
        q = NTL::GenPrime_ZZ(keyLength / 2, err);
        while (p == q) {
            q = NTL::GenPrime_ZZ(keyLength / 2, err);
        }
        NTL::ZZ n = p * q;
        NTL::ZZ phi = (p - 1) * (q - 1);
        if (NTL::GCD(n, phi) == 1) return;
    }
}

Paillier::Paillier() = default;

void Paillier::keygen(long keyLength) {
    /* Paillier parameters generation function. Generates paillier parameters from scrach.
     *
     * Parameters
     * ==========
     * long keyLength: the length of the key
     *
     * =======
     * public key  = (modulus, generator)
     * private key = lambda
     */

//    NTL::SetSeed(NTL::ZZ(0));

    this->keyLength = keyLength;
    GenPrimePair(p, q, keyLength);
    modulus = p * q;                                                        // modulus = p * q
    generator = modulus + 1;                                                // generator = modulus + 1
    lambda = lcm(p - 1, q - 1);                                                    // lamda = lcm(p-1, q-1)
    lambda_power = NTL::PowerMod(generator, lambda, modulus * modulus);
    u = NTL::InvMod(L_function(lambda_power),
                    modulus);                        // u = L((generator^lambda) mod n ^ 2) ) ^ -1 mod modulus

//    random = Gen_Coprime(modulus);
}

NTL::ZZ Paillier::add(const NTL::ZZ &x, const NTL::ZZ &y) const {
    /* Paillier addition function. Computes the sum of x and y.
     *
     * Parameters
     * ==========
     * NTL::ZZ x, y: signed, arbitrary length integers
     *
     * Returns
     * =======
     * NTL:ZZ sum: the sum of x and y
     */
    NTL::ZZ sum = x * y % (modulus * modulus);
    return sum;
}

NTL::ZZ Paillier::mul(const NTL::ZZ &x, const NTL::ZZ &y) const {
    /* Paillier multiplication function. Computes the product of x and y.
     *
     * Parameters
     * ==========
     * NTL::ZZ x, y: signed, arbitrary length integers
     *
     * Returns
     * =======
     * NTL:ZZ sum: the product of x and y
     */
    NTL::ZZ product = PowerMod(x, y, modulus * modulus);
    return product;
}

NTL::ZZ Paillier::encrypt(const NTL::ZZ &message) const {
    /* Paillier encryption function. Takes in a message in F(modulus), and returns a message in F(modulus**2).
     *
     * Parameters
     * ==========
     * NTL::ZZ message : The message to be encrypted.
     *
     * Returns
     * =======
     * NTL:ZZ ciphertext : The encyrpted message.
     */

    NTL::ZZ c = Gen_Coprime(modulus);
    NTL::ZZ ciphertext =
            NTL::PowerMod(generator, message, modulus * modulus) * NTL::PowerMod(c, modulus, modulus * modulus) %
            (modulus * modulus);
    return ciphertext;
}

NTL::ZZ Paillier::decrypt(const NTL::ZZ &ciphertext) const {
    /* Paillier decryption function. Takes in a ciphertext in F(modulus**2), and returns a message in F(modulus).
     *
      * Parameters
     * ==========
     * NTL::ZZ cipertext : The encrypted message.
     *
     * Returns
     * =======
     * NTL::ZZ message : The original message.
     */

    NTL::ZZ deMasked = NTL::PowerMod(ciphertext, lambda, modulus * modulus);
    NTL::ZZ L_deMasked = L_function(deMasked);
    NTL::ZZ message = (L_deMasked * u) % modulus;
    return message;
}
