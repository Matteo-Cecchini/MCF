#include <iostream>

extern "C" { 

long double fibonacci(int n) {
    long double fib = 1, fob = 1, pp; // fib è F_n, fob è F_n-1 
    if (n > 2) {
        for (int i = 2; i < n; i++) {
            pp = fib;
            fib = fib + fob;
            fob = pp;
        }
    }
    return fib/fob;
}

}