#include "blis.h"

packm_thread_info_t* bli_create_packm_thread_info( thread_comm_t* communicator, dim_t tid, dim_t max_threads )
{
    packm_thread_info_t* to_ret = (packm_thread_info_t*) bli_malloc(sizeof(packm_thread_info_t));

    to_ret->communicator = communicator;
    to_ret->tid = tid;
    to_ret->max_threads = max_threads;

    return to_ret;
}

thread_comm_t* bli_packm_communicator( packm_thread_info_t * info )
{
    if( info == NULL )
        return NULL;
    else return info->communicator;
}

dim_t bli_packm_tid( packm_thread_info_t * info )
{
    if( info == NULL )
        return 0;
    else return info->tid;
}

dim_t bli_packm_num_threads( packm_thread_info_t * info )
{
    if( info == NULL ) return 1;
    int nt = info->communicator->num_threads;
    if( nt > info->max_threads ) nt = info->max_threads;
    return nt;
}
