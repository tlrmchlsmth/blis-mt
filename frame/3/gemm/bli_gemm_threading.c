#include "blis.h"

gemm_ker_thread_info_t* bli_create_gemm_ker_thread_info( dim_t l2_tid, dim_t l2_num_threads, dim_t l1_tid, dim_t l1_num_threads, dim_t l0_tid )
{
    gemm_ker_thread_info_t* info = (gemm_ker_thread_info_t*) bli_malloc(sizeof(gemm_ker_thread_info_t));
    info->l2_tid = l2_tid;
    info->l2_num_threads = l2_num_threads;
    info->l1_tid = l1_tid;
    info->l1_num_threads = l1_num_threads;
    info->l0_tid = l0_tid;

    return info;
}

void bli_gemm_ker_thread_info_free( void * info)
{
    gemm_ker_thread_info_t* tofree = (gemm_ker_thread_info_t*) info;
    bli_free( tofree );
}
dim_t bli_gemm_l2_tid( void* info )
{
    if( info == NULL ) return 0;
    else return ((gemm_ker_thread_info_t*)info)->l2_tid;
}
dim_t bli_gemm_l2_num_threads( void* info )
{
    if( info == NULL ) return 0;
    else return ((gemm_ker_thread_info_t*)info)->l2_num_threads;
}
dim_t bli_gemm_l1_tid( void* info )
{
    if( info == NULL ) return 0;
    else return ((gemm_ker_thread_info_t*)info)->l1_tid;
}
dim_t bli_gemm_l1_num_threads( void* info )
{
    if( info == NULL ) return 0;
    else return ((gemm_ker_thread_info_t*)info)->l1_num_threads;
}
dim_t bli_gemm_l0_tid( void* info )
{
    if( info == NULL ) return 0;
    else return ((gemm_ker_thread_info_t*)info)->l0_tid;
}

void bli_gemm_blk_thread_info_free( void * info)
{
    gemm_blk_thread_info_t* tofree = (gemm_blk_thread_info_t*) info;
    bli_free( tofree );
}
dim_t bli_gemm_num_thread_groups( void* info )
{
    if( info == NULL ) return 1;
    else return ((gemm_blk_thread_info_t*)info)->num_groups;
}
dim_t bli_gemm_group_id( void* info )
{
    if( info == NULL ) return 0;
    return ((gemm_blk_thread_info_t*)info)->group_id;
}

bool_t bli_gemm_am_a_master( void* info ) {
    if( info == NULL ) return 1;
    return ((gemm_blk_thread_info_t*)info)->a_id == 0;
}
dim_t bli_gemm_a_num_threads( void* info ) {
    if( info == NULL ) return 1;
    return ((gemm_blk_thread_info_t*)info)->a_comm->num_threads;
}
dim_t bli_gemm_a_id( void* info ) {
    if( info == NULL ) return 0;
    return ((gemm_blk_thread_info_t*)info)->a_id;
}
void bli_gemm_a_barrier( void* info ) {
    if( info == NULL )
        return;
    else bli_barrier( ((gemm_blk_thread_info_t*)info)->a_comm );
}
void* bli_gemm_broadcast_a( void* info, void* to_send) {
    if( info == NULL ) return to_send;
    return bli_broadcast_structure( ((gemm_blk_thread_info_t*)info)->a_comm, ((gemm_blk_thread_info_t*)info)->a_id, to_send );
}

bool_t bli_gemm_am_b_master( void* info ) { 
        if( info == NULL ) return 1;
            return ((gemm_blk_thread_info_t*)info)->b_id == 0;
}
dim_t bli_gemm_b_num_threads( void* info ) { 
        if( info == NULL ) return 1;
            return ((gemm_blk_thread_info_t*)info)->b_comm->num_threads;
}
dim_t bli_gemm_b_id( void* info ) { 
        if( info == NULL ) return 0;
            return ((gemm_blk_thread_info_t*)info)->b_id;
}
void bli_gemm_b_barrier( void* info ) { 
        if( info == NULL )
                    return;
                        else bli_barrier( ((gemm_blk_thread_info_t*)info)->b_comm );
}
void* bli_gemm_broadcast_b( void* info, void* to_send) {
        if( info == NULL ) return to_send;
            return bli_broadcast_structure( ((gemm_blk_thread_info_t*)info)->b_comm, ((gemm_blk_thread_info_t*)info)->b_id, to_send );
}

bool_t bli_gemm_am_c_master( void* info ) {
    if( info == NULL ) return 1;
    return ((gemm_blk_thread_info_t*)info)->c_id == 0;
}
dim_t bli_gemm_c_num_threads( void* info ) {
    if( info == NULL ) return 1;
    return ((gemm_blk_thread_info_t*)info)->c_comm->num_threads;
}
dim_t bli_gemm_c_id( void* info ) {
    if( info == NULL ) return 0;
    return ((gemm_blk_thread_info_t*)info)->c_id;
}
void bli_gemm_c_barrier( void* info ) { 
    if( info == NULL )
        return;
    else bli_barrier( ((gemm_blk_thread_info_t*)info)->c_comm );
}
void* bli_gemm_broadcast_c( void* info, void* to_send) {
    if( info == NULL ) return to_send;
    return bli_broadcast_structure( ((gemm_blk_thread_info_t*)info)->c_comm, ((gemm_blk_thread_info_t*)info)->c_id, to_send );
}

gemm_blk_thread_info_t* bli_create_gemm_blk_thread_info(
        thread_comm_t* a_comm, dim_t a_id,
        thread_comm_t* b_comm, dim_t b_id,
        thread_comm_t* c_comm, dim_t c_id,
        dim_t num_groups, dim_t group_id )
{
    gemm_blk_thread_info_t* info = (gemm_blk_thread_info_t*) bli_malloc(sizeof(gemm_blk_thread_info_t));

    info->a_comm    = a_comm;
    info->a_id      = a_id;
    info->b_comm    = b_comm;
    info->b_id      = b_id;
    info->c_comm    = c_comm;
    info->c_id      = c_id;


    info->num_groups = num_groups;
    info->group_id = group_id;
    return info;
}

