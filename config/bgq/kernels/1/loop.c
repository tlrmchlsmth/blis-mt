#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main(int argc, char * argv[])
{
    int max = argc>1 ? atoi(argv[1]) : 100;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num = omp_get_num_threads();
        for ( int i = tid; i < max; i += num )
            printf("tid = %d i = %d \n", tid, i);
    }

    return 0;
}

