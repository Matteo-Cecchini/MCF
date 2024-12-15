#include <iostream>

extern "C" {

double fibonacci(int n) {
    double fib = 1, fob = 1, pp;
    if (n > 2) {
        for (int i = 2; i != n; i++) {
            pp = fib;
            fib = fib + fob;
            fob = pp;
        }
    }
    return fib/fob;
}

}