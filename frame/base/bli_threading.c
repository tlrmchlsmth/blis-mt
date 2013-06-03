#include "blis.h"

void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    bli_destroy_lock( &communicator->barrier_lock );
}
void bli_setup_communicator( thread_comm_t* communicator, dim_t num_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->num_threads = num_threads;
    communicator->barrier_sense = 0;
    bli_init_lock( &communicator->barrier_lock );
    communicator->barrier_threads_arrived = 0;
}

thread_comm_t* bli_create_communicator( dim_t num_threads )
{
    thread_comm_t* comm = (thread_comm_t*) bli_malloc( sizeof(thread_comm_t) );
    bli_setup_communicator( comm, num_threads );
    return comm;
}

void* bli_broadcast_structure( thread_comm_t* communicator, dim_t id, void* to_send )
{   
    if( communicator == NULL ) return to_send;

    if( id == 0 ) communicator->sent_object = to_send;

    bli_barrier( communicator );
    void * object = communicator->sent_object;
    bli_barrier( communicator );

    return object;
}

void bli_init_lock( lock_t* lock )
{
    omp_init_lock( lock );
}
void bli_destroy_lock( lock_t* lock )
{
    omp_destroy_lock( lock );
}
void bli_set_lock( lock_t* lock )
{
    omp_set_lock( lock );
}
void bli_unset_lock( lock_t* lock )
{
    omp_unset_lock( lock );
}

//barrier routine taken from art of multicore programming or something
void bli_barrier( thread_comm_t* communicator )
{
    if(communicator == NULL)
        return;
    bool_t my_sense = communicator->barrier_sense;
    dim_t my_threads_arrived;

    bli_set_lock(&communicator->barrier_lock);
    my_threads_arrived = communicator->barrier_threads_arrived + 1;
    communicator->barrier_threads_arrived = my_threads_arrived;
    bli_unset_lock(&communicator->barrier_lock);

    if( my_threads_arrived == communicator->num_threads ) {

        bli_set_lock(&communicator->barrier_lock);
        communicator->barrier_threads_arrived = 0;
        communicator->barrier_sense = !communicator->barrier_sense;
        bli_unset_lock(&communicator->barrier_lock);
    }
    else {
        volatile bool_t* listener = &communicator->barrier_sense;
        while( *listener == my_sense ) {}
    }
}



