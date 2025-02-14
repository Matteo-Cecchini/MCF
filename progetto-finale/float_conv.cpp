extern "C" {
#include <cstdlib>  // Per strtod
#include <cstring>  // Per strtok
#include <cctype>   // Per isdigit

double* float_conv(int size, char** mat) {
    /*
    Funzione per convertire array di stringhe in float64.
    ---------------
    Parametri:
        size: la lunghezza dell'array da convertire
        mat: puntatore a puntatori char, ovvero un array di stringhe

    Return:
        puntatore a double con tutte le stringhe convertite. Se la stringa non contiene dati numerici la conversione riporta zero.
    */
    double* res = new double[size];
    for (int i = 0; i < size; i++) {
        char* str = mat[i];
        while (*str && !isdigit(*str)) str++;
        res[i] = strtod(str, nullptr);
    }
    return res;
}
}
