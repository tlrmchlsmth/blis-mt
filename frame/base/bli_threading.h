#ifndef BLIS_THREADING_H
#define BLIS_THREADING_H

typedef omp_lock_t lock_t;

void bli_barrier();
dim_t bli_thread_id();


struct thread_comm_s
{
    void*   sent_object;
    dim_t   num_threads;

    bool_t  barrier_sense;
    lock_t  barrier_lock;
    dim_t   barrier_threads_arrived;
};
typedef struct thread_comm_s thread_comm_t;

struct thread_info_s
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
typedef struct thread_info_s thread_info_t;

void    bli_setup_communicator( thread_comm_t* communicator, dim_t num_threads );

thread_comm_t*    bli_create_communicator( dim_t num_threads );
thread_info_t*    bli_create_thread_info( 
        thread_comm_t* a_comm, dim_t a_id, 
        thread_comm_t* b_comm, dim_t b_id, 
        thread_comm_t* c_comm, dim_t c_id, 
        dim_t num_groups, dim_t group_id);

void*   bli_broadcast_structure( thread_comm_t* communicator, dim_t inside_id, void* to_send );

void*   bli_broadcast_a( thread_info_t* thread_info, void* to_send );
void    bli_a_barrier( thread_info_t* thread_info );
dim_t   bli_a_id( thread_info_t* thread_info );
dim_t   bli_a_num_threads( thread_info_t* thread_info );
bool_t  bli_am_a_master( thread_info_t* info );

void*   bli_broadcast_b( thread_info_t* thread_info, void* to_send );
void    bli_b_barrier( thread_info_t* thread_info );
dim_t   bli_b_id( thread_info_t* thread_info );
dim_t   bli_b_num_threads( thread_info_t* thread_info );
bool_t  bli_am_b_master( thread_info_t* info );

void*   bli_broadcast_c( thread_info_t* thread_info, void* to_send );
void    bli_c_barrier( thread_info_t* thread_info );
dim_t   bli_c_id( thread_info_t* thread_info );
dim_t   bli_c_num_threads( thread_info_t* thread_info );
bool_t  bli_am_c_master( thread_info_t* info );

dim_t   bli_num_thread_groups( thread_info_t* thread_info );
dim_t   bli_group_id( thread_info_t* thread_info );

void    bli_barrier( thread_comm_t* communicator );
void    bli_set_lock( lock_t* lock );
void    bli_unset_lock( lock_t* lock );
void    bli_init_lock( lock_t* lock );
void    bli_destroy_lock( lock_t* lock );

bool_t  bli_am_outside_master( thread_info_t* info );

#endif
