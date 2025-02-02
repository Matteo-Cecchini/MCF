#include <vector>
#include <string>
#include <iostream>
using namespace std;

extern "C" {
void float_conv(int size, char** mat, double* res) {
    char* check;
    for (int i = 0; i != size; i++) {
        do {
            res[i] = strtod((*mat), &check);
            (*mat)++;
        } while (*check != '\0');
        mat++;
    }
}
}