#ifndef BLIS_GEMM_THREADING_H
#define BLIS_GEMM_THREADING_H

struct gemm_ker_thread_info_s
{
    dim_t l2_tid;
    dim_t l2_num_threads;
    dim_t l1_tid;
    dim_t l1_num_threads;
    dim_t l0_tid;

    //Can shove implementation dependent data here 
    //Useful on BG/Q and KNC to synchronize threads that share an L1 cache
    void* other; 
};
typedef struct gemm_ker_thread_info_s gemm_ker_thread_info_t;

gemm_ker_thread_info_t* bli_create_gemm_ker_thread_info( dim_t l2_tid, dim_t l2_num_threads, dim_t l1_tid, dim_t l1_num_threads, dim_t l0_tid );
void  bli_gemm_ker_thread_info_free( void* info);
dim_t bli_gemm_l2_tid( void* thread_info );
dim_t bli_gemm_l2_num_threads( void* thread_info );
dim_t bli_gemm_l1_tid( void* thread_info );
dim_t bli_gemm_l1_num_threads( void* thread_info );
dim_t bli_gemm_l0_tid( void* thread_info );

struct gemm_blk_thread_info_s
{
    thread_comm_t* a_comm;
    dim_t a_id;
    thread_comm_t* b_comm;
    dim_t b_id;
    thread_comm_t* c_comm;
    dim_t c_id;

    dim_t num_groups;
    dim_t group_id;
};
typedef struct gemm_blk_thread_info_s gemm_blk_thread_info_t;

gemm_blk_thread_info_t*    bli_create_gemm_blk_thread_info(
        thread_comm_t* a_comm, dim_t a_id,
        thread_comm_t* b_comm, dim_t b_id,
        thread_comm_t* c_comm, dim_t c_id,
        dim_t num_groups, dim_t group_id );

void  bli_gemm_blk_thread_info_free( void* info);

void*   bli_gemm_broadcast_a( void* thread_info, void* to_send );
void    bli_gemm_a_barrier( void* thread_info );
dim_t   bli_gemm_a_id( void* thread_info );
dim_t   bli_gemm_a_num_threads( void* thread_info );
bool_t  bli_gemm_am_a_master( void* info );

void*   bli_gemm_broadcast_b( void* thread_info, void* to_send );
void    bli_gemm_b_barrier( void* thread_info );
dim_t   bli_gemm_b_id( void* thread_info );
dim_t   bli_gemm_b_num_threads( void* thread_info );
bool_t  bli_gemm_am_b_master( void* info );

void*   bli_gemm_broadcast_c( void* thread_info, void* to_send );
void    bli_gemm_c_barrier( void* thread_info );
dim_t   bli_gemm_c_id( void* thread_info );
dim_t   bli_gemm_c_num_threads( void* thread_info );
bool_t  bli_gemm_am_c_master( void* info );

dim_t   bli_gemm_num_thread_groups( void* thread_info );
dim_t   bli_gemm_group_id( void* thread_info );

#endif
