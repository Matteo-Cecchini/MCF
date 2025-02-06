extern "C" {
#include <cstdlib>  // Per strtod
#include <cstring>  // Per strtok
#include <cctype>   // Per isdigit

double* float_conv(int size, char** mat) {
    double* res = new double[size];
    for (int i = 0; i < size; i++) {
        char* str = mat[i];
        while (*str && !isdigit(*str)) str++;
        res[i] = strtod(str, nullptr);
    }
    return res;
}
}
