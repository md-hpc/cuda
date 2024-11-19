#include <stdio.h>

#define UNIVERSE_SIZE 3
#define T 100
#define DENSITY 10
#define EPSILON 40
#define SIGMA 1
// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define BLOCK_SIZE 256
#define MAX_PARTICLE_ID 1024

struct Particle {
    int particleID;
    float vx;
    float vy;
    float vz;
    float ax;
    float ay;
    float az;
};

struct Cell {
    struct Particle particle_list[];
};

// the meat:
__global__ void simulation(struct Cell cell_list[3][4][4])
{
    // define/assign shared/local memory
    __shared__ struct Cell home_cell_position_cache[];
    __shared__ struct Cell neighbor_cell_position_cache[];
    __shared__ float forces[];

    __syncthreads();

    // iterate through time steps
    for (int i = 0; i < T; ++i) {
        // force computation

        //TODO: (hard) figure out how to assign threads to cells efficiently given numerical parameters
        struct Cell home_cell = cell_list[/* thread indexing magic */]
        struct Cell neighbor_cell/*s*/ = cell_list[/* stronger thread indexing magic */]

        //TODO: (easy) write the rest of the force computation
        /*
            for particle_q in neighbor_cell do
                for particle_p in neighbor_cell do
                    force_computation(q, p);
        */


        // motion update (overwrite all particle data in caches efficiently...)

    }
}

struct Cell initialize_cell_list(struct Cell *cell_list)
{
    //TODO: generate random particle data for now or import
}

int main() 
{
    // usually around 256 per block
    // max 8 blocks per block cluster
   

    // defines grid and block dimensions up to 3 dimensions
    // TODO: (hard) find optimal dimensions
    dim3 threadsPerBlock(/* x */, /* y */, /* z */);
    dim3 numBlocks(/* x */, /* y */, /* z */);

    // initialize particle data for simulation
    struct Cell cell_list[3][4][4];
    initialize_cell_list(&cell_list);

    // the meat of the program, make sure to have o/p of particle data..
    simulation<<<numBlocks, threadsPerBlock>>>(cell_list);

    // TODO (easy) write back to CPU
    // cudaMemcpy();
    // free device memory
    // free host memory

    return 0;
}
