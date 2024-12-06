#include <stdio.h>


// minimum max shared memory size per SM across all architectures is 64K
// minimum max resident block per SM across all architectures is 16
// so worst case, each block will have max 4K shared memory

// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define NUM_PARTICLES 5
#define MAX_PARTICLES_PER_CELL 128

#define CELL_CUTOFF_RADIUS 1f
#define CELL_LENGTH_X 3
#define CELL_LENGTH_Y 3
#define CELL_LENGTH_Z 3

#define TIMESTEPS 1
#define TIMESTEP_DURATION 1                            
#define EPSILON 1f
#define SIGMA 1f
#define LJ_MIN (-4f * 24f * EPSILON / SIGMA * (__powf(7f / 26f, 7f / 6f) - 2f * __powf(7f / 26f, 13f / 6f)))

#define PLUS_1(dimension, length) ((dimension != length - 1) * (dimension + 1))
#define MINUS_1(dimension, length) ((dimension == 0) * length + dimension - 1)

// particle stores coordinates and velocities in x,y,z dimensions
struct Particle {
    int particle_id;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    int new_cell_idx;
};

// cell is an array of particles
struct Cell {
    struct Particle particle_list[MAX_PARTICLES_PER_CELL];
};

// LJ force computation
__device__ float compute_force(float r1, float r2) {
    float r = fabsf(x1 - x2);
    float force = 4 * EPSILON * (6*__powf(SIGMA,6f)/__powf(r,7f) - 12*__powf(SIGMA,12f)/__powf(r,13f));
    if (force < LJ_MIN) {
        force = LJ_MIN;
    }
    return force;
}

// the meat:
__global__ void force_eval(struct Cell *cell_list, float *accelerations)
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
    __shared__ struct Cell home_cell;   // sizeof struct Cell = MAX_PARTICLES_PER_CELL * 28
    __shared__ struct Cell neighbor_cell;
    home_cell.particle_list[threadIdx.x] = cell_list[home_x][home_y][home_z].particle_list[threadIdx.x];
    neighbor_cell.particle_list[threadIdx.x] = cell_list[neighbor_x][neighbor_y][neighbor_z].particle_list[threadIdx.x];

    int reference_particle_id = home_cell.particle_list[threadIdx.x].particle_id;

    // synchronizes threads within a block (all threads must complete tasks)
    __syncthreads();

    if (reference_particle_id != -1) {
        //TODO: (easy) write the rest of the force computation
        for (int i = 0; neighbor_cell.particle_list[i].particle_id != -1 && i < MAX_PARTICLES_PER_CELL; ++i) {
            // boolean expression can be optimized knowing the fact that one dimension of the neighboring half shell is only +1 and not -1
            float neighbor_particle_virtual_x = neighbor_cell.particle_list[i].x + ((home_x - neighbor_x == CELL_LENGTH_X - 1) + (neighbor_x - home_x == CELL_LENGTH_X - 1) * -1) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS);
            float neighbor_particle_virtual_y = neighbor_cell.particle_list[i].y + ((home_y - neighbor_y == CELL_LENGTH_Y - 1) + (neighbor_y - home_y == CELL_LENGTH_Y - 1) * -1) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS);
            float neighbor_particle_virtual_z = neighbor_cell.particle_list[i].z + ((home_z - neighbor_z == CELL_LENGTH_Z - 1) + (neighbor_z - home_z == CELL_LENGTH_Z - 1) * -1) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS);

            // can probably optimize using linear algebras
            atomicAdd(&accelerations[reference_particle_id][0], compute_force(home_cell.particle_list[threadIdx.x].x, neighbor_particle_virtual_x));
            atomicAdd(&accelerations[reference_particle_id][1], compute_force(home_cell.particle_list[threadIdx.x].y, neighbor_particle_virtual_y));
            atomicAdd(&accelerations[reference_particle_id][2], compute_force(home_cell.particle_list[threadIdx.x].z, neighbor_particle_virtual_z));
        }
    }

    __syncthreads();

    // choose one block to "work" on the home cell
    // threads update their associated particle here
    // all particles are still in their original cell
    if (local_idx != 0)
        return;
    home_cell.particle_list[threadIdx.x].vx += accelerations[reference_particle_id] * TIMESTEP_DURATION;
    home_cell.particle_list[threadIdx.x].vy += accelerations[reference_particle_id + 1] * TIMESTEP_DURATION;
    home_cell.particle_list[threadIdx.x].vz += accelerations[reference_particle_id + 2] * TIMESTEP_DURATION;
    home_cell.particle_list[threadIdx.x].x = (home_cell.particle_list[threadIdx.x].x + home_cell.particle_list[threadIdx.x].vx * TIMESTEP_DURATION) - (CELL_LENGTH_X * CELL_CUTOFF_RADIUS) * floor(home_cell.particle_list[threadIdx.x].x / (CELL_LENGTH_X * CELL_CUTOFF_RADIUS));
    home_cell.particle_list[threadIdx.x].y = (home_cell.particle_list[threadIdx.x].y + home_cell.particle_list[threadIdx.x].vy * TIMESTEP_DURATION) - (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS) * floor(home_cell.particle_list[threadIdx.x].y / (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS));
    home_cell.particle_list[threadIdx.x].z = (home_cell.particle_list[threadIdx.x].z + home_cell.particle_list[threadIdx.x].vz * TIMESTEP_DURATION) - (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS) * floor(home_cell.particle_list[threadIdx.x].z / (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS));

    // update global
    cell_list[home_z + home_y * CELL_LENGTH_Z + home_x * CELL_LENGTH_Z * CELL_LENGTH_Y].particle_list[threadIdx.x] = cell.particle_list[threadIdx.x];
}

// update cell lists because particles have moved
__global__ void motion_update(struct Cell *cell_list, float *accelerations)
{
    /*
        1 block per cell
        right now 1 thread per block
        1 thread per particle list
        keeps counter on next free spot on new particle list
        once a -1 in the old particle list is reached, there are no particles to the right
    */
    // get home cell coordinates

    // threadIdx.x is always 0 because we are indexing by blockIdx.x
    int home_x = blockIdx.x % CELL_LENGTH_X;
    int home_y = blockIdx.x / CELL_LENGTH_X % CELL_LENGTH_Y;
    int home_z = blockIdx.x / (CELL_LENGTH_X * CELL_LENGTH_Y) % CELL_LENGTH_Z;

    // declare double-buffer buffer
    struct Cell cell;
    // location of where thread is in buffer
    int free_idx = 0;

    // for every cell,
    // for every particle in that cell,
    // if that particle exists in that cell in the new universe,
    // cell.particle_list[next++] = particle
    // else if particle id == -1
    // cell.particle_list[next].particle_id = -1
    // else continue

    for (int current_cell_idx = 0; current_cell_idx < CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z; ++current_cell_idx) {
        for (int particle_idx = 0; particle_idx < MAX_PARTICLES_PER_CELL && cell_list[current_cell_idx].particle_list[particle_idx].particle_id != -1; ++particle_idx) {
            struct Particle current_particle = cell_list[current_cell_idx].particle_list[particle_idx].x;
            int new_cell_x = current_particle.x / (CELL_LENGTH_X * CELL_CUTOFF_RADIUS);
            int new_cell_y = current_particle.y / (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS);
            int new_cell_z = current_particle.z / (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS);

            if (home_x == new_cell_x && home_y == new_cell_y && home_z == new_cell_z) {
                cell.particle_list[free_idx++] = current_particle;
            }
        }
    }
    cell[free_idx].particle_id = -1;
}

// initialize cells with random particle data
void initialize_cell_list(struct Cell cellList[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z])
{
        // initialize cell list, -1 for empty cell
        memset(cellList, -1, sizeof(cellList));
        for (int i = 0; i < NUM_PARTICLES; ++i) {
                int x = rand() % CELL_LENGTH_X;
                int y = rand() % CELL_LENGTH_Y;
                int z = rand() % CELL_LENGTH_Z;

                // assign random particle data
                struct Particle particle = {
                        .particle_id = i,
                        .x = x * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .y = y * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .z = z * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .vx = 0,
                        .vy = 0,
                        .vz = 0,
                };
                // copy particle to to cell list
                for (int j = 0; j < MAX_PARTICLES_PER_CELL; ++j) {
                    if (cellList[x][y][z].particle_list[j].particle_id == -1) {
                        memcpy(&cellList[x][y][z].particle_list[j], &particle, sizeof(struct Particle));
                        break;
                    }
                }
        }
}

int main() 
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE CELL LIST WITH PARTICLE DATA
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // initialize (or import) particle data for simulation
    struct Cell cell_list[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z];
    initialize_cell_list(&cell_list);
    // device_cell_list stores an array of Cells, where each Cell contains a particle_list
    struct Cell *device_cell_list;
    // cudaMalloc initializes GPU global memory to be used as parameter for GPU kernel
    cudaMalloc(&device_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell));
    cudaMemcpy(device_cell_list, cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE ACCELERATIONS
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
        accelerations stores accelerations (in x y z dimensions) of each particle to be used in motion update.
        - index of accelerations is related to particle_id
        - particle_id * 3 gives index of accelerations for x dimension
        - (particle_id * 3) + 1 gives index of y
        - (particle_id * 3) + 2 gives index of y
    */
    float *accelerations;
    cudaMalloc(&accelerations, MAX_PARTICLES_PER_CELL * 3 * sizeof(float));
    cudaMemset(accelerations, 0, MAX_PARTICLES_PER_CELL * 3 * sizeof(float));


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE PARAMETERS FOR FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // defines block and thread dimensions
    // dim3 is an integer vector type most commonly used to pass the grid and block dimensions in a kernel invocation [X x Y x Z]
    // there are 2^31 blocks in x dimension while y and z have at most 65536 blocks
    dim3 numBlocksForce(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 14);        // (CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 14) x 1 x 1
    dim3 numBlocksMotion(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);            // (CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z) x 1 x 1
    dim3 threadsPerBlockForce(MAX_PARTICLES_PER_CELL);                              // MAX_PARTICLES_PER_CELL x 1 x 1
    dim3 threadsPerBlockMotion(CELL_LENGTH_X, CELL_LENGTH_Y, CELL_LENGTH_Z);  

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // do force evaluation and motion update for each time step
    // steps are separated to ensure threads are synchronized (that force_eval is done)
    // output of force_eval is stores in device_cell_list and accelerations
    for (int t = 0; t < TIMESTEPS; ++t) {
        force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list, accelerations);
        motion_update<<<numBlocksMotion, 1>>>(device_cell_list, accelerations);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  COPY FINAL RESULT BACK TO HOST CPU
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(cell_list, device_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * struct(struct Cell), cudaMemcpyDeviceToHost);
    cudaFree(device_cell_list);

    return 0;
}
