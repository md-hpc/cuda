#include <stdio.h>

// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define NUM_PARTICLES 5
#define MAX_PARTICLES_PER_CELL 120

#define CELL_CUTOFF_RADIUS 1f
#define CELL_LENGTH_X 3
#define CELL_LENGTH_Y 3
#define CELL_LENGTH_Z 3

#define TIME_STEPS 1
#define DT 1                            // amount of time per time step
#define EPSILON 1
#define SIGMA 1

#define PLUS_1(dimension, length) (!(dimension == length - 1) * (dimension + 1))
#define MINUS_1(dimension, length) (!(dimension == 0) * (dimension - 1) + (dimension == 0) * (length - 1))

struct Particle {
    int particleID;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

struct Cell {
    struct Particle particle_list[MAX_PARTICLES_PER_CELL];
};

// the meat:
__global__ void force_eval(struct Cell cell_list[UNIVERSE_SIZE][UNIVERSE_SIZE][UNIVERSE_SIZE])
{
    /*
        1D block array will look like this:
                14               14
        | HNNNNNNNNNNNNN | HNNNNNNNNNNNNN | ... | 
         0               14               28    X*Y*Z*14

        Map one block to a home-neighbor tuple (home cell, neighbor cell)
        Map one thread to a particle index in the home cell, which calculates accelerations in 
        a one to all fashion with the particles in the neighbor cell.

        CAREFUL: one of the home-neighbor tuple will actually be a home-home tuple
    */

    int home_x = blockIdx.x / (14 * CELL_LENGTH_Y * CELL_LENGTH_Z) % CELL_LENGTH_X;
    int home_y = blockIdx.x / (14 * CELL_LENGTH_Z) % CELL_LENGTH_Y;
    int home_z = blockIdx.x / 14 % CELL_LENGTH_Z;

    // trust me on this :)
    int local_idx = blockIdx.x % 14;
    int neighbor_x = (local_idx < 9) * PLUS_1(home_x, CELL_LENGTH_X)
                   + (local_idx >= 9) * home_x;
    int neighbor_y = (local_idx < 3) * MINUS_1(home_y, CELL_LENGTH_Y)
                   + (local_idx >= 3 && local_idx <= 5 || local_idx > 11) * home_y
                   + (local_idx >= 6 && local_idx <= 11) * PLUS_1(home_y, CELL_LENGTH_Y);
    int neighbor_z = (local_idx % 3 == 0) * PLUS_1(home_z, CELL_LENGTH_Z)
                   + (local_idx % 3 == 1) * home_z
                   + (local_idx % 3 == 2) * MINUS_1(home_z, CELL_LENGTH_Z);

    // define and assign shared memory
    __shared__ struct Cell home_cell;
    __shared__ struct Cell neighbor_cell;
    home_cell.particle_list[threadIdx.x] = cell_list[home_x][home_y][home_z].particle_list[threadIdx.x];
    neighbor_cell.particle_list[threadIdx.x] = cell_list[neighbor_x][neighbor_y][neighbor_z].particle_list[threadIdx.x];

    int reference_particle_id = home_cell.particle_list[threadIdx.x].particleID;
    if (reference_particle_id == -1)
        return;

    __syncthreads();

    //TODO: (easy) write the rest of the force computation
    for (int i = 0; neighbor_cell.particle_list[i].particleID != -1 && i < MAX_PARTICLES_PER_CELL; ++i) {
        atomicAdd(&accelerations[reference_particle_id][0], compute_force(home_cell.particle_list[threadIdx.x].x, neighbor_cell.particle_list[i].x));
        atomicAdd(&accelerations[reference_particle_id][1], compute_force(home_cell.particle_list[threadIdx.x].x, neighbor_cell.particle_list[i].x));
        atomicAdd(&accelerations[reference_particle_id][2], compute_force(home_cell.particle_list[threadIdx.x].x, neighbor_cell.particle_list[i].x));
    }
}

struct Cell initialize_cell_list(struct Cell *cell_list)
{
    //TODO: generate random particle data for now or import
}

int main() 
{
    // defines block and thread dimensions
    dim3 numBlocks(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 14);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_CELL);

    // initialize particle data for simulation
    struct Cell cell_list[CELL_LENGTH_X][CELL_LENGTH_Y][CELL_LENGTH_Z];
    initialize_cell_list(&cell_list);

    // the meat of the program, make sure to have o/p of particle data..
    // TODO: (medium) finish up force_eval
    // TODO: (hard) finish up motion_update
    for (int t = 0; t < T; ++t) {
        force_eval<<<numBlocks, threadsPerBlock>>>(cell_list);
        motion_update<<<>>>();
    }

    // TODO (easy) write back to CPU
    // cudaMemcpy();
    // free device memory
    // free host memory

    return 0;
}
