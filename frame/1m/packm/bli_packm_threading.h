#ifndef _BLIS_PACKM_THREADING_H
#define _BLIS_PACKM_THREADING_H

struct packm_thread_info_s
{
    thread_comm_t* communicator;
    dim_t tid;
    dim_t max_threads;
};
typedef struct packm_thread_info_s packm_thread_info_t;

void bli_free_packm_thread_info( packm_thread_info_t* info );
packm_thread_info_t* bli_create_packm_thread_info( thread_comm_t* communicator, dim_t tid, dim_t max_threads );
thread_comm_t* bli_packm_communicator( packm_thread_info_t * info );
dim_t bli_packm_tid( packm_thread_info_t * info );
dim_t bli_packm_num_threads( packm_thread_info_t * info );

#endif
