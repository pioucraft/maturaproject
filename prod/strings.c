#include <stdio.h>

char *oldStr;

char *strtok_multi(char *str, const char *delim) {
    if(delim == NULL) {
        return oldStr;
    }

    char *strtotok;

    if (str == NULL) {
        strtotok = oldStr;
    } else {
        strtotok = str;
    }

    int len = 0;
    int currentDelimPosition = 0;
    int finished = 0;
    oldStr = strtotok;

    while (!finished) {
        if (strtotok[len] == '\0') {
            finished = 1;
        } else if (strtotok[len] == delim[currentDelimPosition]) {
            oldStr++;
            len++;

            currentDelimPosition++;

            if (delim[currentDelimPosition] == '\0') {
                finished = 1;
                strtotok[len - currentDelimPosition] = '\0';
            }
        } else {
            currentDelimPosition = 0;

            oldStr++;
            len++;
        }
    }

    return strtotok;
}
