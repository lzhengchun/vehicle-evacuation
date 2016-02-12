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

#define CUDA_BLOCK_SIZE    32
#define VEHICLE_PER_STEP   1.5
#define EPS                1e-5
#define ENV_DIM_X          200
#define ENV_DIM_Y          200
#define N_ITER             500
#define MAX_CAP            10.f

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
__global__ void evacuation_update(float *p_vcnt_in, float *p_vcnt_out, float *cap, float4 *pturn, 
                                  int Ngx, int Ngy, float * d_halo_sync, curandState_t* states) 
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int g_idy = blockIdx.y*blockDim.y + threadIdx.y;
    int uni_id = g_idy * Ngx + g_idx;
    
    if(g_idx >= Ngx || g_idy >= Ngy)
    {
        return;
    }
    __shared__ float4 io[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2];
    __shared__ float halo_sync[4][CUDA_BLOCK_SIZE];  // order: N -> E -> S -> W
    // use the flag to ignore env edge
    bool update_flag = g_idx >= 1 && g_idx <= Ngx-2 && g_idy >= 1 && g_idy <= Ngy-2;
    
    float cnt_temp = p_vcnt_in[uni_id];
    int idx = threadIdx.x + 1, idy = threadIdx.y + 1;
// 1st step, fill in current [r, c] with # of outing vehicle, (determine # of care go L/R/U/S)
// note that, this is NOT the number of vehicles will be out after the current step
// it depends on the saturation of neighboors, but sure will not be more than the outgoing capacity(depends on speed of vehicle)
    float cnt_out = fminf(VEHICLE_PER_STEP, cnt_temp);
    float cnt_out_bk = cnt_out;
    float4 pturn_c = pturn[uni_id];          // turn probabilities of the cell [i, j]
    io[idy][idx].x = cnt_out * pturn_c.x;    // go north
    io[idy][idx].y = cnt_out * pturn_c.y;    // go east
    io[idy][idx].z = cnt_out * pturn_c.z;    // go south
    io[idy][idx].w = cnt_out * pturn_c.w;    // go west

    // extra work for edge threads, for the halo
    if(threadIdx.x == 0){                        // left halo
        if(update_flag){
            pturn_c = pturn[uni_id-1];
            cnt_out = fminf(VEHICLE_PER_STEP, p_vcnt_in[uni_id-1]);
            io[idy][0].x = cnt_out * pturn_c.x;      // go north
            io[idy][0].y = cnt_out * pturn_c.y;      // go east
            io[idy][0].z = cnt_out * pturn_c.z;      // go south
            io[idy][0].w = cnt_out * pturn_c.w;      // go west    
            halo_sync[3][idy] = io[idy][0].y;        // will be used to computing how many vehicles get accepted by west cell
        }else{
            io[idy][0].x = 0;      // go north
            io[idy][0].y = 0;      // go east
            io[idy][0].z = 0;      // go south
            io[idy][0].w = 0;      // go west     
            halo_sync[3][idy] = 0;          
        }
    }
    if(threadIdx.x == CUDA_BLOCK_SIZE-1){        // right halo
        if(update_flag){
            pturn_c = pturn[uni_id+1];
            cnt_out = fminf(VEHICLE_PER_STEP, p_vcnt_in[uni_id+1]);
            io[idy][CUDA_BLOCK_SIZE+1].x = cnt_out * pturn_c.x;    // go north
            io[idy][CUDA_BLOCK_SIZE+1].y = cnt_out * pturn_c.y;    // go east
            io[idy][CUDA_BLOCK_SIZE+1].z = cnt_out * pturn_c.z;    // go south
            io[idy][CUDA_BLOCK_SIZE+1].w = cnt_out * pturn_c.w;    // go west   
            halo_sync[1][idy] = io[idy][CUDA_BLOCK_SIZE+1].w;
        }else{
            io[idy][CUDA_BLOCK_SIZE+1].x = 0;    // go north
            io[idy][CUDA_BLOCK_SIZE+1].y = 0;    // go east
            io[idy][CUDA_BLOCK_SIZE+1].z = 0;    // go south
            io[idy][CUDA_BLOCK_SIZE+1].w = 0;    // go west   
            halo_sync[1][idy] = 0;            
        }
    }

    if(threadIdx.y == 0){	                     // top halo 
        if(update_flag){
            pturn_c = pturn[uni_id-Ngx];
            cnt_out = fminf(VEHICLE_PER_STEP, p_vcnt_in[uni_id-Ngx]);
            io[0][idx].x = cnt_out * pturn_c.x;    // go north
            io[0][idx].y = cnt_out * pturn_c.y;    // go east
            io[0][idx].z = cnt_out * pturn_c.z;    // go south
            io[0][idx].w = cnt_out * pturn_c.w;    // go west          
            halo_sync[0][idx] = io[0][idx].z;
        }else{
            io[0][idx].x = 0;    // go north
            io[0][idx].y = 0;    // go east
            io[0][idx].z = 0;    // go south
            io[0][idx].w = 0;    // go west          
            halo_sync[0][idx] = 0;
        }
    }    
    if(threadIdx.y == CUDA_BLOCK_SIZE-1){        // bottom halo
        if(update_flag){
            pturn_c = pturn[uni_id+Ngx];
            cnt_out = fminf(VEHICLE_PER_STEP, p_vcnt_in[uni_id+Ngx]);        
            io[CUDA_BLOCK_SIZE+1][idx].x = cnt_out * pturn_c.x;    // go north
            io[CUDA_BLOCK_SIZE+1][idx].y = cnt_out * pturn_c.y;    // go east
            io[CUDA_BLOCK_SIZE+1][idx].z = cnt_out * pturn_c.z;    // go south
            io[CUDA_BLOCK_SIZE+1][idx].w = cnt_out * pturn_c.w;    // go west    
            halo_sync[2][idx] = io[CUDA_BLOCK_SIZE+1][idx].x;      
        }else{
            io[CUDA_BLOCK_SIZE+1][idx].x = 0;    // go north
            io[CUDA_BLOCK_SIZE+1][idx].y = 0;    // go east
            io[CUDA_BLOCK_SIZE+1][idx].z = 0;    // go south
            io[CUDA_BLOCK_SIZE+1][idx].w = 0;    // go west    
            halo_sync[2][idx] = 0;              
        }
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
            case 0:	                             // enter from top 
                if(diff_cap > io[idx][idy-1].z)
                {
                    diff_cap -= io[idx][idy-1].z;
                    io[idx][idy-1].z = 0.f;
                }else{
                    io[idx][idy-1].z -= diff_cap;
                    diff_cap = 0.0;
                }
                break;
            case 1:                              // enter from left
                if(diff_cap > io[idx+1][idy].w)
                {
                    diff_cap -= io[idx+1][idy].w;
                    io[idx+1][idy].w = 0.f;
                }else{
                    io[idx+1][idy].w -= diff_cap;
                    diff_cap = 0.0;
                }
                break;
            case 2:                              // enter from bottom
                if(diff_cap > io[idx][idy+1].x)
                {
                    diff_cap -= io[idx][idy+1].x;
                    io[idx][idy+1].x = 0.f;
                }else{
                    io[idx][idy+1].x -= diff_cap;
                    diff_cap = 0.0;
                }
                break;
            case 3:                              // enter from right
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
    if(update_flag){
        p_vcnt_out[uni_id] = cnt_temp - (cnt_out_bk - io[idy][idx].x - io[idy][idx].y - io[idy][idx].z - io[idy][idx].w) 
                    + (diff_bk - diff_cap);
        __syncthreads();
    }
// 3rd step, process halo synchronization!!!! synchronizing via device global memory    
// to update, we have to know how much vehicle actully went out (get accepted by neighboor)
    int blk_uid = blockIdx.y*gridDim.x + blockIdx.x;
    int id_helper = blk_uid * (4 * CUDA_BLOCK_SIZE);
    if(update_flag && threadIdx.x == 0){                                // left
        id_helper += 3*CUDA_BLOCK_SIZE + threadIdx.y;
        d_halo_sync[id_helper] = halo_sync[3][idy] - io[idy][0].y;      // number of vehicles which actully go out
    }      
    if(update_flag && threadIdx.x == CUDA_BLOCK_SIZE-1){                // right
        id_helper += CUDA_BLOCK_SIZE + threadIdx.y;
        d_halo_sync[id_helper] = halo_sync[1][idy] - io[idy][CUDA_BLOCK_SIZE+1].w;
    }

    if(update_flag && threadIdx.y == 0){                                // top
        id_helper += threadIdx.x;
        d_halo_sync[id_helper] = halo_sync[0][idx] - io[0][idx].z;
    }

    if(update_flag && threadIdx.y == CUDA_BLOCK_SIZE-1){                // bottom
        id_helper += 2*CUDA_BLOCK_SIZE + threadIdx.x;
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
__global__ void evacuation_halo_sync(float *cnt, float *cap, float4 *pturn, 
                                     int Ngx, int Ngy, float * d_halo_sync) 
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
        id_helper += 3*CUDA_BLOCK_SIZE + threadIdx.y;
        cnt[uni_id] -= d_halo_sync[id_helper];  
    }      
    if(threadIdx.x == CUDA_BLOCK_SIZE-1 && blockIdx.x < gridDim.x-1){        // right
        int id_helper = (blockIdx.y*gridDim.x + blockIdx.x + 1) * (4 * CUDA_BLOCK_SIZE);
        id_helper += CUDA_BLOCK_SIZE + threadIdx.y;
        cnt[uni_id] -= d_halo_sync[id_helper]; 
    }

    if(threadIdx.y == 0 && blockIdx.y > 0){                                  // top
        int id_helper = ( (blockIdx.y-1)*gridDim.x + blockIdx.x) * (4 * CUDA_BLOCK_SIZE);
        id_helper += threadIdx.x;
        cnt[uni_id] -= d_halo_sync[id_helper]; 
    }

    if(threadIdx.y == CUDA_BLOCK_SIZE-1 && blockIdx.y < gridDim.y-1){        // bottom
        int id_helper = ( (blockIdx.y+1)*gridDim.x + blockIdx.x) * (4 * CUDA_BLOCK_SIZE);
        id_helper += 2*CUDA_BLOCK_SIZE + threadIdx.x;
        cnt[uni_id] -= d_halo_sync[id_helper]; 
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
    for(int r = 1; r < Ngy-1; r++){
        for(int c = 1; c < Ngx-1; c++){
            int idx = r*Ngx+c;
            p_turn[idx].x = 0.1;
            p_turn[idx].y = 0.7;
            p_turn[idx].z = 0.1;
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
void evacuation_state_init(float *p_cnt, float *p_cap, int Ngx, int Ngy)
{
    for(int r = 1; r < Ngy-1; r++){
        for(int c = 1; c < Ngx-1; c++){
            int idx = r*Ngx+c;
            p_cap[idx] = MAX_CAP;
            p_cnt[idx] = p_cap[idx] * rand() / RAND_MAX;
        }
    }
    // edge
    int idx;
    // first row
    for(int c = 0; c < Ngx; c++){
        p_cap[c] = MAX_CAP;
        p_cnt[c] = 0;
    }
    // left and right
    for(int r = 0; r < Ngy; r++){
        idx = r * Ngx + 0;
        p_cap[idx] = MAX_CAP;
        p_cnt[idx] = 0;
        
        idx = r * Ngx + Ngx-1;
        p_cap[idx] = MAX_CAP;
        p_cnt[idx] = 0;        
    }
    // bottom
    for(int c = 0; c < Ngx; c++){
        idx = (Ngy-1)*Ngx + c;
        p_cap[c] = MAX_CAP;
        p_cnt[c] = 0;
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
            output_file << p_vcnt[idx] << ",";
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
    float *d_helper;                             // order: north -> east -> south -> west
    cudaError_t cuda_error;
    float *h_vcnt = new float[Ngx*Ngy]();
    float *h_vcap = new float[Ngx*Ngy]();
    float4 *h_turn = new float4[Ngx*Ngy]();
    evacuation_field_init(h_turn, Ngx, Ngy);
    evacuation_state_init(h_vcnt, h_vcap, Ngx, Ngy);
    float *d_vcnt_in, *d_vcnt_out, *d_vcap, *p_swap;
    float4 *d_turn;
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    
    cuda_error = cudaMalloc((void**)&d_vcnt_in, sizeof(float)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }    
    cuda_error = cudaMalloc((void**)&d_vcnt_out, sizeof(float)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }
    cuda_error = cudaMalloc((void**)&d_vcap, sizeof(float)*Ngx*Ngy);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }
    cuda_error = cudaMalloc((void**)&d_turn, sizeof(float4)*Ngx*Ngy);
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
    // copy data from host to device
    cuda_error = cudaMemcpy((void *)d_vcnt_in, (void *)h_vcnt, sizeof(float)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }  
    cuda_error = cudaMemcpy((void *)d_vcnt_out, (void *)d_vcnt_in, sizeof(float)*Ngx*Ngy, cudaMemcpyDeviceToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    } 
    cuda_error = cudaMemcpy((void *)d_vcap, (void *)h_vcap, sizeof(float)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }  
    cuda_error = cudaMemcpy((void *)d_turn, (void *)h_turn, sizeof(float4)*Ngx*Ngy, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess){
        cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }    
        
    int helper_size = 4 * CUDA_BLOCK_SIZE * dimGrid.x * dimGrid.y * sizeof(float);
    cuda_error = cudaMalloc((void**)&d_helper, helper_size);
    if (cuda_error != cudaSuccess)
    {
        cout << "CUDA error in cudaMalloc: " << cudaGetErrorString(cuda_error) << endl;
        exit(-1);
    }
    // config to use more shared memory, less L1 cache
    cudaFuncSetCacheConfig(evacuation_update, cudaFuncCachePreferShared);
    
    for(int i = 0; i < N_ITER; i++){
        evacuation_update<<<dimGrid, dimBlock>>>(d_vcnt_in, d_vcnt_out, d_vcap, d_turn, Ngx, Ngy, d_helper, curand_states);
        cuda_error = cudaThreadSynchronize();
        if (cuda_error != cudaSuccess){
            cout << "CUDA error in cudaThreadSynchronize, update: " << cudaGetErrorString(cuda_error) << endl;
            exit(-1);
        } 
        evacuation_halo_sync<<<dimGrid, dimBlock>>>(d_vcnt_out, d_vcap, d_turn, Ngx, Ngy, d_helper);
        cuda_error = cudaThreadSynchronize();
        if (cuda_error != cudaSuccess){
            cout << "CUDA error in cudaThreadSynchronize, sync halo: " << cudaGetErrorString(cuda_error) << endl;
            exit(-1);
        } 
        if(i%50 == 0) {
            cuda_error = cudaMemcpy((void *)h_vcnt, (void *)d_vcnt_out, sizeof(float)*Ngx*Ngy, cudaMemcpyDeviceToHost);
            if (cuda_error != cudaSuccess){
                cout << "CUDA error in cudaMemcpy: " << cudaGetErrorString(cuda_error) << endl;
                exit(-1);
            }  
            write_vehicle_cnt_info(i, h_vcnt, Ngx, Ngy);
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
