//
// Created by liqinbin on 5/26/22.
//

#include "FedTree/Encryption/diffie_hellman.h"
#include <random>
/*
bool is_prime(int n)
{
    // Corner cases
    if (n <= 1)  return false;
    if (n <= 3)  return true;
    if (n%2 == 0 || n%3 == 0) return false;
    for (int i=5; i*i<=n; i=i+6)
        if (n%i == 0 || n%(i+2) == 0)
            return false;
    return true;
}

int fast_power(int x, unsigned int y, int p)
{
    int res = 1;     // Initialize result

    x = x % p; // Update x if it is more than or
    // equal to p

    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res*x) % p;

        // y must be even now
        y = y >> 1; // y = y/2
        x = (x*x) % p;
    }
    return res;
}

// Utility function to store prime factors of a number
void findPrimefactors(unordered_set<int> &s, int n)
{
    // Print the number of 2s that divide n
    while (n%2 == 0)
    {
        s.insert(2);
        n = n/2;
    }

    // n must be odd at this point. So we can skip
    // one element (Note i = i +2)
    for (int i = 3; i <= sqrt(n); i = i+2)
    {
        // While i divides n, print i and divide n
        while (n%i == 0)
        {
            s.insert(i);
            n = n/i;
        }
    }

    // This condition is to handle the case when
    // n is a prime number greater than 2
    if (n > 2)
        s.insert(n);
}

// Function to find smallest primitive root of n
int get_primitive(int n)
{
    unordered_set<int> s;

    // Check if n is prime or not
    if (isPrime(n)==false)
        return -1;

    // Find value of Euler Totient function of n
    // Since n is a prime number, the value of Euler
    // Totient function is n-1 as there are n-1
    // relatively prime numbers.
    int phi = n-1;

    // Find prime factors of phi and store in a set
    findPrimefactors(s, phi);

    // Check for every number from 2 to phi
    for (int r=2; r<=phi; r++)
    {
        // Iterate through all prime factors of phi.
        // and check if we found a power with value 1
        bool flag = false;
        for (auto it = s.begin(); it != s.end(); it++)
        {

            // Check if r^((phi)/primefactors) mod n
            // is 1 or not
            if (power(r, phi/(*it), n) == 1)
            {
                flag = true;
                break;
            }
        }

        // If there was no power with value 1.
        if (flag == false)
            return r;
    }

    // If no primitive root found
    return -1;
}
*/

ZZ toDec(char val){
    if (val=='A' || val=='a') return to_ZZ(10);
    else if(val=='B' || val=='b') return to_ZZ(11);
    else if(val=='C' || val=='c') return to_ZZ(12);
    else if(val=='D' || val=='d') return to_ZZ(13);
    else if(val=='E' || val=='e') return to_ZZ(14);
    else if(val=='F' || val=='f') return to_ZZ(15);
    else return to_ZZ(val-'0');
}

ZZ hexToZZ(string hexVal){
    ZZ val;
    val = to_ZZ(0);	//initialise the value to zero
    double base = 16;
    int j = 0;
    //convert the hex string to decimal string
    for (int i = ((hexVal.length())-1); i > -1; i--){
        val += toDec(hexVal[i])*(to_ZZ((pow(base, j))));
        j++;
    }
    //cout << endl << "The value in decimal is " << val << endl;
    return val;
}

//void DiffieHellman::primegen(){
//    std::random_device rd; // obtain a random number from hardware
//    std::mt19937 gen(rd()); // seed the generator
//    std::uniform_int_distribution<> distr(1e^5, 1e^8); // define the range
//    while(true){
//        p = distr(gen);
//        if(is_prime(p))
//            break;
//    }
//    g = get_primitive(p);

    // from https://datatracker.ietf.org/doc/html/rfc2409#page-22

//}
DiffieHellman::DiffieHellman(){
    //use default value as it does not affect the security
    //from https://datatracker.ietf.org/doc/html/rfc2409#page-22
    p = hexToZZ("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519"
                "B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A89"
                "9FA5AE9F24117C4B1FE649286651ECE65381FFFFFFFFFFFFFFFF");
    g = 2;
}

ZZ DiffieHellman::encrypt(float_type &message, int pid){
    return (message*1e6 + shared_keys[pid])%p;
}


float_type DiffieHellman::decrypt(ZZ &message, int pid){
    return (float_type) NTL::to_long((message - shared_keys[pid])%p) / 1e6;
}

void DiffieHellman::generate_public_key(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, 100); // define the range
    secret = distr(gen);
    public_key = PowerMod(g, secret, p);
    return;
};

void DiffieHellman::init_variables(int n_parties){
    this->n_parties = n_parties;
    other_public_keys.SetLength(n_parties);
    shared_keys.SetLength(n_parties);
    encrypted_noises.SetLength(n_parties);
    generated_noises.resize(n_parties);
    received_encrypted_noises.SetLength(n_parties);
    decrypted_noises.resize(n_parties);
};

void DiffieHellman::compute_shared_keys(){
    for(int i = 0; i < other_public_keys.length(); i++){
        if(i!=pid) {
            shared_keys[i] = PowerMod(other_public_keys[i], secret, p);
        }
    }

}

void DiffieHellman::generate_noises(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1e6, 1e9); // define the range
    for(int i = 0; i < n_parties; i++){
        if(i!=pid) {
            generated_noises[i] = (float_type) distr(gen) / 1e6;
            encrypted_noises[i] = encrypt(generated_noises[i], i);
        }
    }
    return;
}

void DiffieHellman::decrypt_noises(){
    for(int i = 0; i < n_parties; i++){
        if(i != pid)
            decrypted_noises[i] = decrypt(received_encrypted_noises[i], i);
    }
    return;
}