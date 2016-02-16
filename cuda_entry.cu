/*
 *  file name: cuda_entry.cu
 *
 * 
 * nvcc -c cuda_entry.c -lcurand
 * or compile against the static cuRAND library
 * nvcc -c cuda_entry.c -lcurand_static -lculibos
 */
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <vector_functions.h>
#include <cstdlib>
#include <fstream>

#define CUDA_BLOCK_SIZE    16
#define VEHICLE_PER_STEP   1.5
#define EPS                1e-5
#define ENV_DIM_X          100
#define ENV_DIM_Y          100
#define N_ITER             4000
#define MAX_CAP            10.f
#define TL_PERIOD          5                          // traffic light period, # of steps, must be integer
using namespace std;
/*
***********************************************************************************************************
* 
*                    Global Variables
*                   
***********************************************************************************************************
*/
// curand library uses curandState_t to keep track of the seed value 
// we will store a random state for every thread 

// duplicate the last one to avoid random number generater curand_uniform generate exactly 1.0
__constant__  unsigned char order[25][4] = {{0,1,2,3}, {0,1,3,2}, {0,2,1,3}, {0,2,3,1}, 
                                            {0,3,1,2}, {0,3,2,1}, {1,0,2,3}, {1,0,3,2}, 
                                            {1,2,0,3}, {1,2,3,0}, {1,3,0,2}, {1,3,2,0}, 
                                            {2,0,1,3}, {2,0,3,1}, {2,1,0,3}, {2,1,3,0}, 
                                            {2,3,0,1}, {2,3,1,0}, {3,0,1,2}, {3,0,2,1}, 
                                            {3,1,0,2}, {3,1,2,0}, {3,2,0,1}, {3,2,1,0}, {3,2,1,0}};
/*
***********************************************************************************************************
* func   name: curand_init_all
* description: this GPU kernel function is used to initialize the random states
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
__global__ void curand_init_all(unsigned int seed, curandState_t* states, int Ngx, int Ngy) {
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int g_idy = blockIdx.y*blockDim.y + threadIdx.y;
    int uni_id = g_idy * Ngx + g_idx;
    if(g_idx >= Ngx || g_idy >= Ngy)
    {
        return;
    }
    curand_init(seed,       /* the seed can be the same for each core, here we pass the time in from the CPU */
                uni_id,     /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0,          /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[uni_id]);
}

__device__ float4 operator*(const float4 & a, const float4 & b) {

    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
/*
***********************************************************************************************************
* func   name: evacuation_update
* description: this GPU kernel function is used to foward the simulation one step. each thread will process
               one cell, map was divided to cell as an intersection model (corresponding turn probabilities 
               were set as zero if the cell is not a real intersection in reality).
* parameters :
*             p_vcnt_in   : device pointer to vehicle counter as input, .x -> horizontal, .y -> vertical
*             p_vcnt_out  : device pointer to vehicle counter as output, .x -> horizontal, .y -> vertical
*             cap         : device pointer to cell capacity
*             pturn       : device pointer to turn probabilities
*             d_tl        : device pointer to traffic light information
*             Ngx         : map size, X dimension
*             Ngy         : map size, Y dimension
*             d_halo_sync : pointer for thread block border synchronization
*             ts          : current time step
*             states      : cuda random state
* note: counter represents number of vehicles who want to go n(x), e(y), s(z), w(w)
* return: none
* note:   cuda vec 4 type: x->north; y->east; z->south; w->west; 
***********************************************************************************************************
*/
__global__ void evacuation_update(float4 *p_vcnt_in, float4 *p_vcnt_out, float *cap, float4 *pturn, 
                                  uchar2 *d_tl, int Ngx, int Ngy, float * d_halo_sync, int time_step, 
                                  curandState_t* states) 
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int g_idy = blockIdx.y*blockDim.y + threadIdx.y;
    int uni_id = g_idy * Ngx + g_idx;
    
    if(g_idx >= Ngx || g_idy >= Ngy)
    {
        return;
    }
    __shared__ float4    io[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2];
    __shared__ float4 io_bk[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2];
    __shared__ float halo_sync[4][CUDA_BLOCK_SIZE];  // order: N -> E -> S -> W
    // use the flag to ignore outmost layer
    // bool update_flag = g_idx >= 1 && g_idx <= Ngx-2 && g_idy >= 1 && g_idy <= Ngy-2;
    bool exit_flag   = g_idx > 80 && g_idx <= Ngx-1 && g_idy == Ngy-1;
    exit_flag = exit_flag || (g_idx == Ngx-1 && g_idy >= 50 && g_idy <= 70);
    
    bool tl_hor = (time_step - (int)tl_info.x) % TL_PERIOD < tl_info.y; // current traffic light
   
    float4 cnt_temp = p_vcnt_in[uni_id];
    int idx = threadIdx.x + 1, idy = threadIdx.y + 1;
/// 1st step, fill in current [r, c] with # of outing vehicle, (determine # of care go L/R/U/S)
// note that, this is NOT the number of vehicles will be out after the current step
// it depends on the saturation of neighboors, but sure will not be more than the outgoing capacity(depends on speed of vehicle)
    
    uchar2 tl_info = d_tl[uni_id];           	                 // traffic light information
    if( tl_hor ){                                                // horizontal light
        io[idy][idx].x = 0.f;                                    // go north
        io[idy][idx].y = fminf(VEHICLE_PER_STEP, cnt_temp.y);    // go east        
        io[idy][idx].z = 0.f;                                    // go south
        io[idy][idx].w = fminf(VEHICLE_PER_STEP, cnt_temp.w);    // go west     
    }
    else{	                                                     // vertical light 
        io[idy][idx].x = fminf(VEHICLE_PER_STEP, cnt_temp.x);    // go north
        io[idy][idx].y = 0.f;                                    // go east
        io[idy][idx].z = fminf(VEHICLE_PER_STEP, cnt_temp.z);    // go south
        io[idy][idx].w = 0.f;                                    // go west
    }
    io_bk[idy][idx] = io[idy][idx];                             // back up the number for calculating difference

    // extra work for edge threads, for the halo, only one direction needs determine
    if(threadIdx.x == 0){                            	        // left halo
        tl_info = d_tl[uni_id-1];                               // traffic light
        float4 v_cnt = p_vcnt_in[uni_id-1];
        if( (time_step - (int)tl_info.x) % TL_PERIOD < tl_info.y ){  // horizontal light
            io[idy][0].x = 0.f;                                      // go north
            io[idy][0].y = fminf(VEHICLE_PER_STEP, v_cnt.y);         // go east        
            io[idy][0].z = 0.f;                                      // go south
            io[idy][0].w = fminf(VEHICLE_PER_STEP, v_cnt.w);         // go west    
            
        }else{	                                                     // vertical light 
            io[idy][0].x = fminf(VEHICLE_PER_STEP, v_cnt.x);         // go north
            io[idy][0].y = 0.f;                                      // go east       
            io[idy][0].z = fminf(VEHICLE_PER_STEP, v_cnt.z);         // go south
            io[idy][0].w = 0.f;                                      // go west        
        }
        io_bk[idy][0] = io[idy][0];  
        halo_sync[3][idy] = io[idy][0].y;        	                 // will be used to computing how many vehicles get accepted by west cell
    }

    if(threadIdx.x == CUDA_BLOCK_SIZE-1){                       // right halo
        tl_info = d_tl[uni_id+1];                               // traffic light
        float4 v_cnt = p_vcnt_in[uni_id+1];
        if( (time_step - (int)tl_info.x) % TL_PERIOD < tl_info.y ){  // horizontal light
            io[idy][CUDA_BLOCK_SIZE+1].x = 0.f;                                    // go north
            io[idy][CUDA_BLOCK_SIZE+1].y = fminf(VEHICLE_PER_STEP, v_cnt.y);       // go east        
            io[idy][CUDA_BLOCK_SIZE+1].z = 0.f;                                    // go south
            io[idy][CUDA_BLOCK_SIZE+1].w = fminf(VEHICLE_PER_STEP, v_cnt.w);       // go west    
            
        }else{	                                                     // vertical light 
            io[idy][CUDA_BLOCK_SIZE+1].x = fminf(VEHICLE_PER_STEP, v_cnt.x);       // go north
            io[idy][CUDA_BLOCK_SIZE+1].y = 0.f;                                    // go east       
            io[idy][CUDA_BLOCK_SIZE+1].z = fminf(VEHICLE_PER_STEP, v_cnt.z);       // go south
            io[idy][CUDA_BLOCK_SIZE+1].w = 0.f;                                    // go west        
        }
        io_bk[idy][CUDA_BLOCK_SIZE+1] = io[idy][CUDA_BLOCK_SIZE+1];  
        halo_sync[1][idy] = io[idy][CUDA_BLOCK_SIZE+1].z;        	           // will be used to computing how many vehicles get accepted by west cell
    }

    if(threadIdx.y == 0){                                       // top halo
        tl_info = d_tl[uni_id-Ngx];                             // traffic light
        float4 v_cnt = p_vcnt_in[uni_id-Ngx];
        if( (time_step - (int)tl_info.x) % TL_PERIOD < tl_info.y ){  // horizontal light
            io[0][idx].x = 0.f;                                    // go north
            io[0][idx].y = fminf(VEHICLE_PER_STEP, v_cnt.y);       // go east        
            io[0][idx].z = 0.f;                                    // go south
            io[0][idx].w = fminf(VEHICLE_PER_STEP, v_cnt.w);       // go west    
            
        }else{	                                                   // vertical light 
            io[0][idx].x = fminf(VEHICLE_PER_STEP, v_cnt.x);       // go north
            io[0][idx].y = 0.f;                                    // go east       
            io[0][idx].z = fminf(VEHICLE_PER_STEP, v_cnt.z);       // go south
            io[0][idx].w = 0.f;                                    // go west        
        }
        io_bk[0][idx] = io[0][idx]; 
        halo_sync[0][idx] = io[0][idx].z;        	             // will be used to computing how many vehicles get accepted by west cell
    }
            
    if(threadIdx.y == CUDA_BLOCK_SIZE-1){                                       // bottom halo
        tl_info = d_tl[uni_id+Ngx];                               // traffic light
        float4 v_cnt = p_vcnt_in[uni_id+Ngx];
        if( (time_step - (int)tl_info.x) % TL_PERIOD < tl_info.y ){  // horizontal light
            io[CUDA_BLOCK_SIZE+1][idx].x = 0.f;                                    // go north
            io[CUDA_BLOCK_SIZE+1][idx].y = fminf(VEHICLE_PER_STEP, v_cnt.y);       // go east        
            io[CUDA_BLOCK_SIZE+1][idx].z = 0.f;                                    // go south
            io[CUDA_BLOCK_SIZE+1][idx].w = fminf(VEHICLE_PER_STEP, v_cnt.w);       // go west    
            
        }else{	                                                   // vertical light 
            io[CUDA_BLOCK_SIZE+1][idx].x = fminf(VEHICLE_PER_STEP, v_cnt.x);       // go north
            io[CUDA_BLOCK_SIZE+1][idx].y = 0.f;                                    // go east       
            io[CUDA_BLOCK_SIZE+1][idx].z = fminf(VEHICLE_PER_STEP, v_cnt.z);       // go south
            io[CUDA_BLOCK_SIZE+1][idx].w = 0.f;                                    // go west        
        }
        io_bk[CUDA_BLOCK_SIZE+1][idx] = io[CUDA_BLOCK_SIZE+1][idx]; 
        halo_sync[2][idx] = io[CUDA_BLOCK_SIZE+1][idx].x;      	                  // will be used to computing how many vehicles get accepted by west cell
    }
      
    // then wait untill all the threads in the same thread block finish their outgoing computing processing
    __syncthreads();  

/// 2nd step, process incoming vehicles, it will update outgoing requests of neighboors. 
    float4 diff_cap, diff_bk;                     // the capacity of incoming vehicles 
    diff_cap.x =  = cap[uni_id]/4.f - cnt_temp.x; 
    diff_cap.y =  = cap[uni_id]/4.f - cnt_temp.y; 
    diff_cap.z =  = cap[uni_id]/4.f - cnt_temp.z; 
    diff_cap.w =  = cap[uni_id]/4.f - cnt_temp.w; 
    diff_bk = diff_cap;                                  // save the capacity for computing how many vehicles entered at the end
    // priority ? random
    // returns a random number between 0.0 and 1.0 following a uniform distribution.
    
    float4 pturn_c = pturn[uni_id];          // turn probabilities of the cell [i, j]
    int rnd = (unsigned char)( curand_uniform(&states[uni_id])*24 ); 
    for (int i=0; i<4 && (diff_cap.x > EPS || diff_cap.y > EPS); i++)
    {
        switch(order[rnd][i])
        {
            case 0:	                             // enter from top 
                if(io[idy-1][idx].z > 0){           
                    float4 in_distr = pturn_c * io[idy-1][idx].z;  // incoming distribution
                }else{
                    break;
                }                                                                          
                io[idy-1][idx].z = in_distr.x + in_distr.y + in_distr.z + in_distr.w;
                break;
            case 1:                              // enter from left
                if(io[idy][idx-1].y > 0){           
                    float4 in_distr = pturn_c * io[idy][idx-1].y;  // incoming distribution
                }else{
                    break;
                }                                                                                          
                io[idy][idx-1].y = in_distr.x + in_distr.y + in_distr.z + in_distr.w;
                break;
            case 2:                              // enter from bottom
                if(io[idy+1][idx].x > 0){           
                    float4 in_distr = pturn_c * io[idy+1][idx].x;  // incoming distribution
                }else{
                    break;
                }                                                                                        
                io[idy+1][idx].x = in_distr.x + in_distr.y + in_distr.z + in_distr.w;
                break;
            case 3:                              // enter from right
                if(io[idy][idx+1].w > 0){           
                    float4 in_distr = pturn_c * io[idy][idx+1].w;  // incoming distribution
                }else{
                    break;
                }                                                                                        
                io[idy][idx+1].w = in_distr.x + in_distr.y + in_distr.z + in_distr.w;            
                break;                                                            
        }
        // to north
        if(diff_cap.x > in_distr.x){
            diff_cap.x -= in_distr.x;
            in_distr.x = 0.f;
        }else{
            in_distr.x -= diff_cap.x;
            diff_cap.x = 0.f;
        }     
        // to east       
        if(diff_cap.y > in_distr.y){
            diff_cap.y -= in_distr.y;
            in_distr.y = 0.f;
        }else{
            in_distr.y -= diff_cap.y;
            diff_cap.y = 0.f;
        }  
        // to south
        if(diff_cap.z > in_distr.z){
            diff_cap.z -= in_distr.z;
            in_distr.z = 0.f;
        }else{
            in_distr.z -= diff_cap.z;
            diff_cap.z = 0.f;
        }  
        // to west
        if(diff_cap.w > in_distr.w){
            diff_cap.w -= in_distr.w;
            in_distr.w = 0.f;
        }else{
            in_distr.w -= diff_cap.w;
            diff_cap.w = 0.f;
        }  
        // write back these who did not get received due to saturation
        switch(order[rnd][i])
        {
            case 0:	                             // enter from top                                                                       
                io[idy-1][idx].z = in_distr.x + in_distr.y + in_distr.z + in_distr.w;
                break;
            case 1:                              // enter from left                                                                                        
                io[idy][idx-1].y = in_distr.x + in_distr.y + in_distr.z + in_distr.w;
                break;
            case 2:                              // enter from bottom                                                                                    
                io[idy+1][idx].x = in_distr.x + in_distr.y + in_distr.z + in_distr.w;
                break;
            case 3:                              // enter from right                                                                                     
                io[idy][idx+1].w = in_distr.x + in_distr.y + in_distr.z + in_distr.w;            
                break;                                                            
        }                      
    } 
    __syncthreads();
// add saturated vehicle back to counter, pre_cnt - (want_go - saturated) + incoming(in_cap - in_cap_left)
    if(!exit_flag){
        p_vcnt_out[uni_id].x = cnt_temp.x - (io_bk[idy][idx].x - io[idy][idx].x) + (diff_bk.x - diff_cap.x);
        p_vcnt_out[uni_id].y = cnt_temp.y - (io_bk[idy][idx].y - io[idy][idx].y) + (diff_bk.y - diff_cap.y);
        p_vcnt_out[uni_id].z = cnt_temp.z - (io_bk[idy][idx].z - io[idy][idx].z) + (diff_bk.z - diff_cap.z);
        p_vcnt_out[uni_id].w = cnt_temp.w - (io_bk[idy][idx].w - io[idy][idx].w) + (diff_bk.w - diff_cap.w);
        __syncthreads();
    }
/// 3rd step, process halo synchronization!!!! synchronizing via device global memory    
// to update, we have to know how much vehicle actully went out (get accepted by neighboor)
    int blk_uid = blockIdx.y*gridDim.x + blockIdx.x;
    int id_helper_st = blk_uid * (4 * CUDA_BLOCK_SIZE);                 // start address in current block
    if(threadIdx.x == 0){                                // left
        int id_helper = id_helper_st + 3*CUDA_BLOCK_SIZE + threadIdx.y;
        d_halo_sync[id_helper] = halo_sync[3][idy] - io[idy][0].y;      // number of vehicles which actully go out
    }      
    if(threadIdx.x == CUDA_BLOCK_SIZE-1){                // right
        int id_helper = id_helper_st + CUDA_BLOCK_SIZE + threadIdx.y;
        d_halo_sync[id_helper] = halo_sync[1][idy] - io[idy][CUDA_BLOCK_SIZE+1].w;
    }

    if(threadIdx.y == 0){                                // top
        int id_helper = id_helper_st + threadIdx.x;
        d_halo_sync[id_helper] = halo_sync[0][idx] - io[0][idx].z;
    }

    if(threadIdx.y == CUDA_BLOCK_SIZE-1){                // bottom
        int id_helper = id_helper_st + 2*CUDA_BLOCK_SIZE + threadIdx.x;
        d_halo_sync[id_helper] = halo_sync[2][idx] - io[CUDA_BLOCK_SIZE+1][idx].x;
    }     
}
/*
***********************************************************************************************************
* func   name: evacuation_halo_sync
* description: this GPU kernel function is used to sync cuda block edge.
* parameters :
*             none
* return: none
* note:   cuda vec 4 type: x->north; y->east; z->south; w->west; 
***********************************************************************************************************
*/
__global__ void evacuation_halo_sync(float *cnt, int Ngx, int Ngy, float * d_halo_sync) 
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int g_idy = blockIdx.y*blockDim.y + threadIdx.y;
    int uni_id = g_idy * Ngx + g_idx;   
    if(g_idx >= Ngx || g_idy >= Ngy)
    {
        return;
    }    
    
    if(threadIdx.x == 0 && blockIdx.x > 0){                                  // left
        int id_helper = (blockIdx.y*gridDim.x + blockIdx.x - 1) * (4 * CUDA_BLOCK_SIZE);
        id_helper += 1*CUDA_BLOCK_SIZE + threadIdx.y;
        cnt[uni_id].x -= d_halo_sync[id_helper];  
    }      
    if(threadIdx.x == CUDA_BLOCK_SIZE-1 && blockIdx.x < gridDim.x-1){        // right
        int id_helper = (blockIdx.y*gridDim.x + blockIdx.x + 1) * (4 * CUDA_BLOCK_SIZE);
        id_helper += 3*CUDA_BLOCK_SIZE + threadIdx.y;
        cnt[uni_id].x -= d_halo_sync[id_helper]; 
    }

    if(threadIdx.y == 0 && blockIdx.y > 0){                                  // top
        int id_helper = ( (blockIdx.y-1)*gridDim.x + blockIdx.x) * (4 * CUDA_BLOCK_SIZE);
        id_helper += 2*CUDA_BLOCK_SIZE + threadIdx.x;
        cnt[uni_id].y -= d_halo_sync[id_helper]; 
    }

    if(threadIdx.y == CUDA_BLOCK_SIZE-1 && blockIdx.y < gridDim.y-1){        // bottom
        int id_helper = ( (blockIdx.y+1)*gridDim.x + blockIdx.x) * (4 * CUDA_BLOCK_SIZE);
        id_helper += threadIdx.x;
        cnt[uni_id].y -= d_halo_sync[id_helper]; 
    }
}
/*
***********************************************************************************************************
* func   name: evacuation_cuda_init
* description: initialize cuda related variable/environment, this function should be called in 
               model initialization
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void evacuation_cuda_init(int Ngx, int Ngy, curandState_t* curand_states){
    int nthread = Ngx * Ngy;
    // allocate space on the GPU for the random states 
    cudaMalloc((void**) &curand_states, nthread * sizeof(curandState_t));
    
    // Launch configuration:
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    
    // invoke the GPU to initialize all of the random states 
    curand_init_all<<<dimGrid, dimBlock>>>(time(0), curand_states, Ngx, Ngy);
}
/*
***********************************************************************************************************
* func   name: evacuation_3D_gpu_finalize
* description: release allocated resource in cuda runtime, this function should be called 
               in model finalize func.
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void evacuation_cuda_finalize()
{
    //cudaFree(curand_states);
}

/*
***********************************************************************************************************
* func   name: evacuation_field_init
* description: initialize the field, i.e., initialize all the turn probabilities 
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void evacuation_field_init(float4 *p_turn, int Ngx, int Ngy)
{
    for(int r = 0; r < Ngy; r++){
        for(int c = 0; c < Ngx; c++){
            int idx = r*Ngx+c;
            p_turn[idx].x = 0.1;
            p_turn[idx].y = 0.4;
            p_turn[idx].z = 0.4;
            p_turn[idx].w = 0.1;
        }
    }
}
/*
***********************************************************************************************************
* func   name: evacuation_state_init
* description: initialize the state, i.e., initialize number of vehicles in each of the cells, and capacity 
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void evacuation_state_init(float2 *p_cnt, float *p_cap, uchar2 *h_tl, int Ngx, int Ngy)
{
    for(int r = 1; r < Ngy-1; r++){
        for(int c = 1; c < Ngx-1; c++){
            int idx = r*Ngx+c;
            p_cap[idx] = MAX_CAP;
            p_cnt[idx].x = p_cap[idx] * rand() / RAND_MAX / 4.f;
            p_cnt[idx].y = p_cap[idx] * rand() / RAND_MAX / 4.f;
            p_cnt[idx].z = p_cap[idx] * rand() / RAND_MAX / 4.f;
            p_cnt[idx].w = p_cap[idx] * rand() / RAND_MAX / 4.f;
        }
    }
    // edge
    int idx;
    // first row
    for(int c = 0; c < Ngx; c++){
        p_cap[c] = MAX_CAP;
        p_cnt[c].x = 0;
        p_cnt[c].y = 0;
    }
    // left and right
    for(int r = 0; r < Ngy; r++){
        idx = r * Ngx + 0;
        p_cap[idx] = MAX_CAP;
        p_cnt[idx].x = 0;
        p_cnt[idx].y = 0;
        
        idx = r * Ngx + Ngx-1;
        p_cap[idx] = MAX_CAP;
        p_cnt[idx].x = 0;
        p_cnt[idx].y = 0;        
    }
    // bottom
    for(int c = 0; c < Ngx; c++){
        idx = (Ngy-1)*Ngx + c;
        p_cap[idx] = MAX_CAP;
        p_cnt[idx].x = 0;
        p_cnt[idx].y = 0;
    }    
    // traffic offset 
    
    for(int r = 0; r < Ngy; r++){
        for(int c = 0; c < Ngx; c++){
            int idx = r*Ngx+c;
            // offset
            h_tl[idx].x = rand() % 5; 
            // pulse wideth for horizontal direction, associates with turn probabilities
            h_tl[idx].y = 1 + rand() % (TL_PERIOD - 1);     
        }
    }
}
/*
***********************************************************************************************************
* func   name: write_vehicle_cnt_info
* description: write results to file for visualizing
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void write_vehicle_cnt_info(int time_step, float * p_vcnt, int Ngx, int Ngy)
{
    ofstream output_file;
    char filename[100];
    sprintf( filename, "vehicle-cnt-info-ts-%d.txt", time_step);
    output_file.open(filename);
    for(int r = 0; r < Ngy; r++){
        for(int c = 0; c < Ngx; c++){
            int idx = r*Ngx+c;
            output_file << p_vcnt[idx].x + p_vcnt[idx].y +  p_vcnt[idx].z +  p_vcnt[idx].w << ",";
        }
        output_file << endl;
    }    
    output_file.close();
}
/*
***********************************************************************************************************
* func   name: write_vehicle_cnt_info
* description: write results to file for visualizing
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void write_halo_sync(int time_step, float * p_halo_sync, int n_block)
{
    ofstream output_file;
    char filename[100];
    sprintf( filename, "halo-sync-ts-%d.txt", time_step);
    output_file.open(filename);
    for(int b = 0; b < n_block; b++){
        for(int h = 0; h < 4*CUDA_BLOCK_SIZE; h++){
            int idx = b*4*CUDA_BLOCK_SIZE + h;       
            output_file << p_halo_sync[idx] << ",";
            if((h+1) % CUDA_BLOCK_SIZE == 0){
                output_file << "||";
            }             
        }
        output_file << endl;
    }    
    output_file.close();
}
/*
***********************************************************************************************************
* func   name: main
* description: main entry of the model implementation
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
int main()
{
    int Ngx = ENV_DIM_X + 2, Ngy = ENV_DIM_Y + 2;
    // this device memory is used for sync block halo, i.e., halo evacuation
    float *d_helper;                                  // order: north -> east -> south -> west
    cudaError_t cuda_error;
    float *h_vcnt    = new float4[Ngx*Ngy]();         // host memory for vehicle counter
    float *h_vcap    = new float[Ngx*Ngy]();          // host memory for vehicle capacity in each of the cells
    float4 *h_turn   = new float4[Ngx*Ngy]();         // host memory for turn probabilities
    uchar2 *h_tlinfo = new uchar2[Ngx*Ngy]();         // host memory for traffic light time offset, and pulse wideth for horizontal
    evacuation_field_init(h_turn, Ngx, Ngy);	      // initialize turn probabilities (the field) 
    evacuation_state_init(h_vcnt, h_vcap, h_tlinfo, Ngx, Ngy);  // initialize vehicle counters and cell capacity
    
    // device memory for counter (as input), counter (as output), capacity, temporary for swaping
    float *d_vcnt_in, *d_vcnt_out, *d_vcap, *p_swap;
    float4 *d_turn;	                             	  // device memory for turn probabilities 
    uchar2 *d_tlinfo;
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    // allocate device memory for vehicle counters as input
    cuda_error = cudaMalloc((void**)&d_vcnt_in, sizeof(float4)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }    
    // allocate device memory for vehicle counters as output
    cuda_error = cudaMalloc((void**)&d_vcnt_out, sizeof(float4)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }
    // allocate device memory for storing cell capacity information
    cuda_error = cudaMalloc((void**)&d_vcap, sizeof(float)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }
    // allocate device memory for saving vehicle turn probabilities
    cuda_error = cudaMalloc((void**)&d_turn, sizeof(float4)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }     
    // allocate device memory for saving traffic light configuration
    cuda_error = cudaMalloc((void**)&d_tlinfo, sizeof(uchar2)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }     
    // initialize random state
    curandState_t* curand_states;
    int nthread = Ngx * Ngy;           // just keep the same as the number of cells
    // allocate space on the GPU for the random states 
    cuda_error = cudaMalloc((void**) &curand_states, nthread * sizeof(curandState_t));
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }    
    // Launch configuration:  
    // invoke the GPU to initialize all of the random states 
    curand_init_all<<<dimGrid, dimBlock>>>(time(0), curand_states, Ngx, Ngy);
    cuda_error = cudaThreadSynchronize();
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaThreadSynchronize, random initialize: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }    
    // copy counter initial values from host to device
    cuda_error = cudaMemcpy((void *)d_vcnt_in, (void *)h_vcnt, sizeof(float4)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }  
    cuda_error = cudaMemcpy((void *)d_vcnt_out, (void *)d_vcnt_in, sizeof(float4)*Ngx*Ngy, cudaMemcpyDeviceToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    } 
    // copy cell capacity information from host to device
    cuda_error = cudaMemcpy((void *)d_vcap, (void *)h_vcap, sizeof(float)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }  
    // copy vehicle turn probabilities from host to device
    cuda_error = cudaMemcpy((void *)d_turn, (void *)h_turn, sizeof(float4)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }    
    // copy traffic light configuration from host to device
    cuda_error = cudaMemcpy((void *)d_tlinfo, (void *)h_tlinfo, sizeof(uchar2)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }         
    int helper_size = 4 * CUDA_BLOCK_SIZE * dimGrid.x * dimGrid.y * sizeof(float);
    // allocate device memory to synchronizing block border counters
    cuda_error = cudaMalloc((void**)&d_helper, helper_size);
    if (cuda_error != cudaSuccess)
    {
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }
    // allocate host memory for checking (debug) thread block border synchronizing operation
    float *h_halo_sync = new float[4 * CUDA_BLOCK_SIZE * dimGrid.x * dimGrid.y]();
    cudaMemcpy((void *)d_helper, (void *)h_halo_sync, 4*CUDA_BLOCK_SIZE * dimGrid.x * dimGrid.y * sizeof(float), cudaMemcpyHostToDevice);
    // config to use more shared memory, less L1 cache
    cudaFuncSetCacheConfig(evacuation_update, cudaFuncCachePreferShared);
    write_vehicle_cnt_info(0, h_vcnt, Ngx, Ngy);  // initial state
    
    for(int i = 0; i < N_ITER; i++){
        evacuation_update<<<dimGrid, dimBlock>>>(d_vcnt_in, d_vcnt_out, d_vcap, d_turn, d_tlinfo, Ngx, Ngy, d_helper, i, curand_states);
        cuda_error = cudaThreadSynchronize();
        if (cuda_error != cudaSuccess){
            cout << "CUDA error in cudaThreadSynchronize, update: " << cudaGetErrorString(cuda_error) << endl;
            exit(-1);
        } 
	    // synchronizing thread block (subdomain) border counters
        evacuation_halo_sync<<<dimGrid, dimBlock>>>(d_vcnt_out, Ngx, Ngy, d_helper);
        cuda_error = cudaThreadSynchronize();
        if (cuda_error != cudaSuccess){
            cout << "CUDA error in cudaThreadSynchronize, sync halo: " << cudaGetErrorString(cuda_error) << endl;
            exit(-1);
        } 
        if((i+1)%100 == 0) {
            cuda_error = cudaMemcpy((void *)h_vcnt, (void *)d_vcnt_out, sizeof(float)*Ngx*Ngy, cudaMemcpyDeviceToHost);
            if (cuda_error != cudaSuccess){
                cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
                exit(-1);
            }  
            write_vehicle_cnt_info(i+1, h_vcnt, Ngx, Ngy);
        }
        p_swap = d_vcnt_in;
        d_vcnt_in = d_vcnt_out;
        d_vcnt_out = p_swap;
    }
    cudaThreadSynchronize();
    cuda_error = cudaMemcpy((void *)h_vcnt, (void *)d_vcnt_in, sizeof(float)*Ngx*Ngy, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }  
    write_vehicle_cnt_info(N_ITER, h_vcnt, Ngx, Ngy);
    
    delete h_vcnt;
    delete h_vcap;
    delete h_turn;
    cudaFree(d_turn);
    cudaFree(d_vcnt_in);
    cudaFree(d_vcnt_out);
    cudaFree(d_vcap);
    cudaFree(d_helper);
}
