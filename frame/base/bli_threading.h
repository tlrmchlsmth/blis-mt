#ifndef BLIS_THREADING_H
#define BLIS_THREADING_H

typedef omp_lock_t lock_t;

struct thread_comm_s
{
    void*   sent_object;
    dim_t   num_threads;

    bool_t  barrier_sense;
    lock_t  barrier_lock;
    dim_t   barrier_threads_arrived;
};
typedef struct thread_comm_s thread_comm_t;

void    bli_setup_communicator( thread_comm_t* communicator, dim_t num_threads );
thread_comm_t*    bli_create_communicator( dim_t num_threads );

void*   bli_broadcast_structure( thread_comm_t* communicator, dim_t inside_id, void* to_send );

void    bli_barrier( thread_comm_t* communicator );
void    bli_set_lock( lock_t* lock );
void    bli_unset_lock( lock_t* lock );
void    bli_init_lock( lock_t* lock );
void    bli_destroy_lock( lock_t* lock );

#endif
