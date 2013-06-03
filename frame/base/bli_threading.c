#include "blis.h"


bool_t bli_am_b_master( thread_info_t* info ) {
    if( info == NULL ) return 1;
    return info->b_id == 0;
}
dim_t bli_b_num_threads( thread_info_t* info ) {
    if( info == NULL ) return 1;
    return info->b_comm->num_threads;
}
dim_t bli_b_id( thread_info_t* info ) {
    if( info == NULL ) return 0;
    return info->b_id;
}
void bli_b_barrier( thread_info_t* info ) {
    if( info == NULL )
        return;
    else bli_barrier( info->b_comm );
}
void* bli_broadcast_b( thread_info_t* info, void* to_send) {
    if( info == NULL ) return to_send;
    return bli_broadcast_structure( info->b_comm, info->b_id, to_send );
}


bool_t bli_am_c_master( thread_info_t* info ) {
    if( info == NULL ) return 1;
    return info->c_id == 0;
}
dim_t bli_c_num_threads( thread_info_t* info ) {
    if( info == NULL ) return 1;
    return info->c_comm->num_threads;
}
dim_t bli_c_id( thread_info_t* info ) {
    if( info == NULL ) return 0;
    return info->c_id;
}
void bli_c_barrier( thread_info_t* info ) {
    if( info == NULL )
        return;
    else bli_barrier( info->c_comm );
}
void* bli_broadcast_c( thread_info_t* info, void* to_send) {
    if( info == NULL ) return to_send;
    return bli_broadcast_structure( info->c_comm, info->c_id, to_send );
}

bool_t bli_am_a_master( thread_info_t* info ) {
    if( info == NULL ) return 1;
    return info->a_id == 0;
}
dim_t bli_a_num_threads( thread_info_t* info ) {
    if( info == NULL ) return 1;
    return info->a_comm->num_threads;
}
dim_t bli_a_id( thread_info_t* info ) {
    if( info == NULL ) return 0;
    return info->a_id;
}
void bli_a_barrier( thread_info_t* info ) {
    if( info == NULL )
        return;
    else bli_barrier( info->a_comm );
}
void* bli_broadcast_a( thread_info_t* info, void* to_send) {
    if( info == NULL ) return to_send;
    return bli_broadcast_structure( info->a_comm, info->a_id, to_send );
}



dim_t bli_num_thread_groups( thread_info_t* info )
{
    if( info == NULL ) return 1;
    else return info->num_groups;
}
dim_t bli_group_id( thread_info_t* info )
{
    if( info == NULL ) return 0;
    return info->group_id;
}

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
thread_info_t* bli_create_thread_info( 
        thread_comm_t* a_comm, dim_t a_id,
        thread_comm_t* b_comm, dim_t b_id,
        thread_comm_t* c_comm, dim_t c_id,
        dim_t num_groups, dim_t group_id )
{
    thread_info_t* info = (thread_info_t*) bli_malloc(sizeof(thread_info_t));

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



