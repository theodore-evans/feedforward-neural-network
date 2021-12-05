#include <cstdlib>
#include "util.h"

double doubleRand() // returns a random number between -1 and 1
{
    return 2 * ((double)rand() / ((double)RAND_MAX + 1.0)) - 1.0;
}
