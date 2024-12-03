#include <stdio.h>

// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define NUM_PARTICLES 5
#define MAX_PARTICLES_PER_CELL 128

#define CELL_CUTOFF_RADIUS 1f
#define CELL_LENGTH_X 3
#define CELL_LENGTH_Y 3
#define CELL_LENGTH_Z 3

#define TIMESTEPS 1
#define TIMESTEP_DURATION 1                            
#define EPSILON 1
#define SIGMA 1

#define PLUS_1(dimension, length) ((dimension != length - 1) * (dimension + 1))
#define MINUS_1(dimension, length) ((dimension == 0) * length + dimension - 1)

// particle stores coordinates and velocities in x,y,z dimensions
struct Particle {
    int particleId;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

// cell is an array of particles
struct Cell {
    struct Particle particle_list[MAX_PARTICLES_PER_CELL];
};

// force computation
__device__ float compute_force(float x1, float x2) {
    float force = 10;
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
    __shared__ struct Cell home_cell;
    __shared__ struct Cell neighbor_cell;
    home_cell.particle_list[threadIdx.x] = cell_list[home_x][home_y][home_z].particle_list[threadIdx.x];
    neighbor_cell.particle_list[threadIdx.x] = cell_list[neighbor_x][neighbor_y][neighbor_z].particle_list[threadIdx.x];

    int reference_particle_id = home_cell.particle_list[threadIdx.x].particleId;
    if (reference_particle_id == -1)
        return;

    __syncthreads();

    //TODO: (easy) write the rest of the force computation
    for (int i = 0; neighbor_cell.particle_list[i].particleId != -1 && i < MAX_PARTICLES_PER_CELL; ++i) {
        // boolean expression can be optimized knowing the fact that one dimension of the neighboring half shell is only +1 and not -1
        float neighbor_particle_virtual_x = neighbor_cell.particle_list[i].x + ((home_x - neighbor_x == CELL_LENGTH_X - 1) + (neighbor_x - home_x == CELL_LENGTH_X - 1) * -1) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS);
        float neighbor_particle_virtual_y = neighbor_cell.particle_list[i].y + ((home_y - neighbor_y == CELL_LENGTH_Y - 1) + (neighbor_y - home_y == CELL_LENGTH_Y - 1) * -1) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS);
        float neighbor_particle_virtual_z = neighbor_cell.particle_list[i].z + ((home_z - neighbor_z == CELL_LENGTH_Z - 1) + (neighbor_z - home_z == CELL_LENGTH_Z - 1) * -1) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS);

        // can probably optimize using linear algebras
        atomicAdd(&accelerations[reference_particle_id][0], compute_force(home_cell.particle_list[threadIdx.x].x, neighbor_particle_virtual_x));
        atomicAdd(&accelerations[reference_particle_id][1], compute_force(home_cell.particle_list[threadIdx.x].y, neighbor_particle_virtual_y));
        atomicAdd(&accelerations[reference_particle_id][2], compute_force(home_cell.particle_list[threadIdx.x].z, neighbor_particle_virtual_z));
    }

    // particle update here, all particles are still in their original cell
}

__global__ void motion_update(struct Cell *cell_list, float *accelerations)
{
    // think about particles moving from cell to cell
    int cell_x = blockIdx.x % CELL_LENGTH_X;
    int cell_y = blockIdx.x / CELL_LENGTH_X % CELL_LENGTH_Y;
    int cell_z = blockIdx.x / (CELL_LENGTH_X * CELL_LENGTH_Y) % CELL_LENGTH_Z;

    __shared__ struct Cell cell;
    cell.particle_list[threadIdx.x] = cell_list[cell_x + cell_y * CELL_LENGTH_X + cell_z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[threadIdx.x];

    int particleId = cell.particle_list[threadIdx.x].particleId;

    // python: cell_from_position = lambda r: linear_idx(*[floor(x/CUTOFF)%UNIVERSE_SIZE for x in r])
    cell.particle_list[threadIdx.x].vx += accelerations[particleId] * TIMESTEP_DURATION;
    cell.particle_list[threadIdx.x].vy += accelerations[particleId + 1] * TIMESTEP_DURATION;
    cell.particle_list[threadIdx.x].vz += accelerations[particleId + 2] * TIMESTEP_DURATION;
    cell.particle_list[threadIdx.x].x = (cell.particle_list[threadIdx.x].x + cell.particle_list[threadIdx.x].vx * TIMESTEP_DURATION) - (CELL_LENGTH_X * CELL_CUTOFF_RADIUS) * floor(cell.particle_list[threadIdx.x].x / (CELL_LENGTH_X * CELL_CUTOFF_RADIUS));
    cell.particle_list[threadIdx.x].y = (cell.particle_list[threadIdx.x].y + cell.particle_list[threadIdx.x].vy * TIMESTEP_DURATION) - (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS) * floor(cell.particle_list[threadIdx.x].y / (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS));
    cell.particle_list[threadIdx.x].z = (cell.particle_list[threadIdx.x].z + cell.particle_list[threadIdx.x].vz * TIMESTEP_DURATION) - (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS) * floor(cell.particle_list[threadIdx.x].z / (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS));

    cell_list[cell_x + cell_y * CELL_LENGTH_X + cell_z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[threadIdx.x] = cell.particle_list[threadIdx.x];

    // put above code in force_eval. this code is supposed to 
    // update cell list with updated particles
    // one block per cell
    // one thread per cell not equal to block's cell
    // that thread loops over each particle and copies it to home cell's particle list

    accelerations[particleId] = 0;
    accelerations[particleId + 1] = 0;
    accelerations[particleId + 2] = 0;
}

void initialize_cell_list(struct Cell cellList[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z])
{
        // initialize cell list, -1 for empty cell
        memset(cellList, -1, sizeof(cellList));
        for (int i = 0; i < NUM_PARTICLES; ++i) {
                int x = rand() % CELL_LENGTH_X;
                int y = rand() % CELL_LENGTH_Y;
                int z = rand() % CELL_LENGTH_Z;

                struct Particle particle = {
                        .particleId = i,
                        .x = x * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .y = y * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .z = z * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .vx = 0,
                        .vy = 0,
                        .vz = 0,
                };
                // copy particle to 
                for (int j = 0; j < MAX_PARTICLES_PER_CELL; ++j) {
                    if (cellList[x][y][z].particle_list[j].particleId == -1) {
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
        - index of accelerations is related to particleId
        - particleId * 3 gives index of accelerations for x dimension
        - (particleId * 3) + 1 gives index of y
        - (particleId * 3) + 2 gives index of y
    */
    float *accelerations;
    cudaMalloc(&accelerations, MAX_PARTICLES_PER_CELL * 3 * sizeof(float));
    cudaMemset(accelerations, 0, MAX_PARTICLES_PER_CELL * 3 * sizeof(float));


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE PARAMETERS FOR FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // defines block and thread dimensions
    // dim3 is an integer vector type most commonly used to pass the grid and block dimensions in a kernel invocation [X x Y x Z]
    dim3 numBlocksForce(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 14);    // (CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 14) x 1 x 1
    dim3 numBlocksMotion(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);        // (CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z) x 1 x 1
    dim3 threadsPerBlock(MAX_PARTICLES_PER_CELL);                               // MAX_PARTICLES_PER_CELL x 1 x 1

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // do force evaluation and motion update for each time step
    // steps are separated to ensure threads are synchronized (that force_eval is done)
    // output of force_eval is stores in device_cell_list and accelerations
    for (int t = 0; t < TIMESTEPS; ++t) {
        force_eval<<<numBlocksForce, threadsPerBlock>>>(device_cell_list, accelerations);
        motion_update<<<numBlocksMotion, threadsPerBlock>>>(device_cell_list, accelerations);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  COPY FINAL RESULT BACK TO HOST CPU
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(cell_list, device_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * struct(struct Cell), cudaMemcpyDeviceToHost);
    cudaFree(device_cell_list);

    return 0;
}
