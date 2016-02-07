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

typedef unsigned char uchar;

#define CUDA_BLOCK_SIZE    16
#define VEHICLE_PER_STEP   1.5
#define EPS                1e-5
/*
***********************************************************************************************************
* 
*                    Global Variables
*                   
***********************************************************************************************************
*/
// curand library uses curandState_t to keep track of the seed value 
// we will store a random state for every thread 
curandState_t* curand_states;
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

/*
***********************************************************************************************************
* func   name: evacuation_update
* description: this GPU kernel function is used to foward the simulation one step. each thread will process
               one cell, map was divided to cell as an intersection model (corresponding turn probabilities 
               were set as zero if the cell is not a real intersection in reality).
* parameters :
*             none
* return: none
* note:   cuda vec 4 type: x->north; y->east; z->south; w->west; 
***********************************************************************************************************
*/
__global__ void evacuation_update(float *cnt, float *cap, float4 *pturn, 
                                  int Ngx, int Ngy, int b2r_i, curandState_t* states) 
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int g_idy = blockIdx.y*blockDim.y + threadIdx.y;
    int uni_id = g_idy * Ngx + g_idx;
    if(g_idx >= Ngx || g_idy >= Ngy)
    {
        return;
    }
    __shared__ float4 io[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2];
    __shared__ float halo_sync[4][CUDA_BLOCK_SIZE];  // order, N -> E -> S -> W
    
    float cnt_temp = cnt[uni_id];
    int idx = threadIdx.x + 1, idy = threadIdx.y + 1;
// 1st step, fill in current [r, c] with # of outing vehicle, (determine # of care go L/R/U/S)
// note that, this is NOT the number of vehicles will be out after the current step
// it depends on the saturation of neighboors, but sure will not be more than the outgoing capacity(depends on speed of vehicle)
    float cnt_out = fminf(VEHICLE_PER_STEP, cnt_temp);
    float4 pturn_c = pturn[uni_id];          // turn probabilities of the cell [i, j]
    io[idy][idx].x = cnt_out * pturn_c.x;    // go north
    io[idy][idx].y = cnt_out * pturn_c.y;    // go east
    io[idy][idx].z = cnt_out * pturn_c.z;    // go south
    io[idy][idx].w = cnt_out * pturn_c.w;    // go west

    // extra work for edge threads, for the halo
    if(idx == 0){
        pturn_c = pturn[uni_id-1];
        cnt_out = fminf(VEHICLE_PER_STEP, cnt[uni_id-1]);
        io[idy][0].x = cnt_out * pturn_c.x;    // go north
        io[idy][0].y = cnt_out * pturn_c.y;    // go east
        io[idy][0].z = cnt_out * pturn_c.z;    // go south
        io[idy][0].w = cnt_out * pturn_c.w;    // go west    
        halo_sync[3][idy] = io[idy][0].y;      // will be used to computing how many vehicles get accepted by west cell
    }
    if(idx == CUDA_BLOCK_SIZE-1){
        pturn_c = pturn[uni_id+1];
        cnt_out = fminf(VEHICLE_PER_STEP, cnt[uni_id+1]);
        io[idy][CUDA_BLOCK_SIZE+1].x = cnt_out * pturn_c.x;    // go north
        io[idy][CUDA_BLOCK_SIZE+1].y = cnt_out * pturn_c.y;    // go east
        io[idy][CUDA_BLOCK_SIZE+1].z = cnt_out * pturn_c.z;    // go south
        io[idy][CUDA_BLOCK_SIZE+1].w = cnt_out * pturn_c.w;    // go west   
        halo_sync[1][idy] = io[idy][CUDA_BLOCK_SIZE+1].w;
    }

    if(idy == 0){
        pturn_c = pturn[uni_id-Ngx];
        cnt_out = fminf(VEHICLE_PER_STEP, cnt[uni_id-Ngx]);
        io[0][idx].x = cnt_out * pturn_c.x;    // go north
        io[0][idx].y = cnt_out * pturn_c.y;    // go east
        io[0][idx].z = cnt_out * pturn_c.z;    // go south
        io[0][idx].w = cnt_out * pturn_c.w;    // go west          
        halo_sync[0][idx] = io[0][idx].z;
    }    
    if(idy == CUDA_BLOCK_SIZE-1){
        pturn_c = pturn[uni_id+Ngx];
        cnt_out = fminf(VEHICLE_PER_STEP, cnt[uni_id+Ngx]);        
        io[CUDA_BLOCK_SIZE+1][idx].x = cnt_out * pturn_c.x;    // go north
        io[CUDA_BLOCK_SIZE+1][idx].y = cnt_out * pturn_c.y;    // go east
        io[CUDA_BLOCK_SIZE+1][idx].z = cnt_out * pturn_c.z;    // go south
        io[CUDA_BLOCK_SIZE+1][idx].w = cnt_out * pturn_c.w;    // go west    
        halo_sync[2][idx] = io[CUDA_BLOCK_SIZE+1][idx].x;      
    }
    // then wait untill all the threads in the sam thread block finish their outgoing conut processing
    __syncthreads();  
// 2nd step, process incoming vehicles, it will update outgoing requests of neighboors. 
    float diff_cap = cap[uni_id] - cnt_temp;                   // the capacity of incoming vehicles 
    float diff_bk = diff_cap;                                  // save the capacity for computing how many vehicles entered at the end
    /// priority ? random
    // returns a random number between 0.0 and 1.0 following a uniform distribution.
    int rnd = (unsigned char)( curand_uniform(&states[uni_id])*24 ); 
    for (int i=0; i<4 && diff_cap > EPS; i++)
    {
        switch(order[rnd][i])
        {
            case 0:
                if(diff_cap > io[idx][idy-1].z)
                {
                    diff_cap -= io[idx][idy-1].z;
                    io[idx][idy-1].z = 0.f;
                }else{
                    io[idx][idy-1].z -= diff_cap;
                    diff_cap = 0.0;
                }
                break;
            case 1:
                if(diff_cap > io[idx+1][idy].w)
                {
                    diff_cap -= io[idx+1][idy].w;
                    io[idx+1][idy].w = 0.f;
                }else{
                    io[idx+1][idy].w -= diff_cap;
                    diff_cap = 0.0;
                }
                break;
            case 2:
                if(diff_cap > io[idx][idy+1].x)
                {
                    diff_cap -= io[idx][idy+1].x;
                    io[idx][idy+1].x = 0.f;
                }else{
                    io[idx][idy+1].x -= diff_cap;
                    diff_cap = 0.0;
                }
                break;
            case 3:
                if(diff_cap > io[idx-1][idy].y)
                {
                    diff_cap -= io[idx-1][idy].y;
                    io[idx-1][idy].y = 0.f;
                }else{
                    io[idx-1][idy].y -= diff_cap;
                    diff_cap = 0.0;
                }
                break;                                                            
        }
    } 

    __syncthreads();
// add saturated vehicle back to counter, pre_cnt - (want_go - saturated) + incoming(in_cap - in_cap_left)
    cnt[uni_id] = cnt_temp - (cnt_out - io[idy][idx].x - io[idy][idx].y - io[idy][idx].z - io[idy][idx].w) 
                + (diff_bk - diff_cap);
   __syncthreads();
   
// 3rd step, process halo synchronization!!!! Wrong!!! You can not guarantee cross thread block sync!!! Change device memory belong to other block is not thread safe
    // to update, we have to know how much vehicle actully went out (get accepted by neighboor)
    if(idx == 0){
        cnt[uni_id-1] -= (halo_sync[3][idy] - io[idy][0].y);
    }      
    if(idx == CUDA_BLOCK_SIZE-1){
        cnt[uni_id+1] -= (halo_sync[1][idy] - io[idy][CUDA_BLOCK_SIZE+1].w);
    }

    if(idy == 0){
        cnt[uni_id-Ngx] -= (halo_sync[0][idx] - io[0][idx].z);
    }

    if(idy == CUDA_BLOCK_SIZE-1){
        cnt[uni_id+Ngx] -= (halo_sync[2][idx] - io[CUDA_BLOCK_SIZE+1][idx].x);
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
void evacuation_cuda_init(int nthread, int Ngx, int Ngy){
    int nthread = Ngx * Ngy;
    // allocate space on the GPU for the random states 
    cudaMalloc((void**) &curand_states, nthread * sizeof(curandState_t));
    
    // Launch configuration:
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    
    // invoke the GPU to initialize all of the random states 
    curand_init_all<<<grid, block>>>(time(0), curand_states);
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
    cudaFree(curand_states);
}
/*
***********************************************************************************************************
* func   name: evacuation_cuda_main
* description: main entry of the model implementation, this function will be called from B2R every R 
               simulation time steps.
* parameters :
*             none
* return: none
***********************************************************************************************************
*/
void evacuation_cuda_main(CELL_DT * h_in, int b2r_R, int b2r_D, int Ngx, int Ngy){
    // Launch configuration:
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    
    evacuation_update<<<grid, block>>>(curand_states, gpu_nums);
}
