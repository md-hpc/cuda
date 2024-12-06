#include <stdio.h>
#include <stdlib.h>
#include <curand.h>


// minimum max shared memory size per SM across all architectures is 64K
// minimum max resident block per SM across all architectures is 16
// so worst case, each block will have max 4K shared memory

// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define _NUM_PARTICLES 5
#define _MAX_PARTICLES_PER_CELL 128

#define _CELL_CUTOFF_RADIUS 1.0f
#define _CELL_LENGTH_X 3
#define _CELL_LENGTH_Y 3
#define _CELL_LENGTH_Z 3

#define _TIMESTEPS 500
#define _TIMESTEP_DURATION 1                            
#define _EPSILON 1.0
#define _SIGMA 1.0
#define _LJ_MIN (-4.0 * 24.0 * _EPSILON / _SIGMA * (powf(7.0 / 26.0, 7.0 / 6.0) - 2.0 * powf(7.0 / 26.0, 13.0 / 6.0)))

#define _PLUS_1(dimension, length) ((dimension != length - 1) * (dimension + 1))
#define _MINUS_1(dimension, length) ((dimension == 0) * length + dimension - 1)

// particle stores coordinates and velocities in x,y,z dimensions
struct Particle {
    int particle_id;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

// cell is an array of particles
struct Cell {
    struct Particle particle_list[_MAX_PARTICLES_PER_CELL];
};

// LJ force computation
// can probably optimize using linear algebras
__device__ float compute_acceleration(float r1, float r2) {
    float r = fabsf(r1 - r2);
    if (r == 0)
        return 0;

    float force = 4 * _EPSILON * (6 * powf(_SIGMA, 6.0) / powf(r, 7.0) - 12 * powf(_SIGMA, 12.0) / powf(r, 13.0));
    if (force < _LJ_MIN) {
        force = _LJ_MIN;
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

    // find hcell coordinate based off of block index
    int home_x = blockIdx.x / (14 * _CELL_LENGTH_Y * _CELL_LENGTH_Z) % _CELL_LENGTH_X;
    int home_y = blockIdx.x / (14 * _CELL_LENGTH_Z) % _CELL_LENGTH_Y;
    int home_z = blockIdx.x / 14 % _CELL_LENGTH_Z;

    // trust me on this :)
    // branchless programming
    int local_idx = blockIdx.x % 14;
    int neighbor_x = (local_idx < 9) * _PLUS_1(home_x, _CELL_LENGTH_X)
                   + (local_idx >= 9) * home_x;
    int neighbor_y = (local_idx < 3) * _MINUS_1(home_y, _CELL_LENGTH_Y)
                   + (local_idx >= 3 && local_idx <= 5 || local_idx > 11) * home_y
                   + (local_idx >= 6 && local_idx <= 11) * _PLUS_1(home_y, _CELL_LENGTH_Y);
    int neighbor_z = (local_idx % 3 == 0) * _PLUS_1(home_z, _CELL_LENGTH_Z)
                   + (local_idx % 3 == 1) * home_z
                   + (local_idx % 3 == 2) * _MINUS_1(home_z, _CELL_LENGTH_Z);

    // define and assign shared memory
    // lowk don't need shared
    __shared__ struct Cell home_cell;
    __shared__ struct Cell neighbor_cell;
    home_cell.particle_list[threadIdx.x] = cell_list[home_z + home_y * _CELL_LENGTH_Z + home_x * _CELL_LENGTH_Z * _CELL_LENGTH_Y].particle_list[threadIdx.x];
    neighbor_cell.particle_list[threadIdx.x] = cell_list[neighbor_z + neighbor_y * _CELL_LENGTH_Z + neighbor_x * _CELL_LENGTH_Z * _CELL_LENGTH_Y].particle_list[threadIdx.x];

    // set the particle thread is assigned to from particle from hcell
    int reference_particle_id = home_cell.particle_list[threadIdx.x].particle_id;

    // synchronizes threads within a block (all threads must complete tasks)
    __syncthreads();

    // if particle exists loop through every particle in ncell particle list
    if (reference_particle_id != -1) {
        //TODO: (easy) write the rest of the force computation
        for (int i = 0; i < _MAX_PARTICLES_PER_CELL && neighbor_cell.particle_list[i].particle_id != -1; ++i) {
            int neighbor_particle_id = neighbor_cell.particle_list[i].particle_id;
            // boolean expression can be optimized knowing the fact that one dimension of the neighboring half shell is only +1 and not -1
            // for periodic boundary condition
            float neighbor_particle_virtual_x = neighbor_cell.particle_list[i].x + ((home_x - neighbor_x == _CELL_LENGTH_X - 1) + (neighbor_x - home_x == _CELL_LENGTH_X - 1) * -1) * (_CELL_LENGTH_X * _CELL_CUTOFF_RADIUS);
            float neighbor_particle_virtual_y = neighbor_cell.particle_list[i].y + ((home_y - neighbor_y == _CELL_LENGTH_Y - 1) + (neighbor_y - home_y == _CELL_LENGTH_Y - 1) * -1) * (_CELL_LENGTH_Y * _CELL_CUTOFF_RADIUS);
            float neighbor_particle_virtual_z = neighbor_cell.particle_list[i].z + ((home_z - neighbor_z == _CELL_LENGTH_Z - 1) + (neighbor_z - home_z == _CELL_LENGTH_Z - 1) * -1) * (_CELL_LENGTH_Z * _CELL_CUTOFF_RADIUS);

            // computed accelerations
            float ax = compute_acceleration(home_cell.particle_list[threadIdx.x].x, neighbor_particle_virtual_x);
            float ay = compute_acceleration(home_cell.particle_list[threadIdx.x].y, neighbor_particle_virtual_y);
            float az = compute_acceleration(home_cell.particle_list[threadIdx.x].z, neighbor_particle_virtual_z);

            // add home particle accelerations
            atomicAdd(&accelerations[reference_particle_id], ax);
            atomicAdd(&accelerations[reference_particle_id + 1], ay);
            atomicAdd(&accelerations[reference_particle_id + 2], az);

            // if not home cell, update the neighbor particle to be -(hcell acceleration) due to N3L
            if (!(home_x == neighbor_x && home_y == neighbor_y && home_z == neighbor_z)) {
                atomicAdd(&accelerations[neighbor_particle_id], -ax);
                atomicAdd(&accelerations[neighbor_particle_id + 1], -ay);
                atomicAdd(&accelerations[neighbor_particle_id + 2], -az);
            }
        }
    }
}

__global__ void particle_update(struct Cell *cell_list, float *accelerations)
{
    // 1 block -> 1 cell
    // 1 thread -> 1 particle

    int reference_particle_id = cell_list[blockIdx.x].particle_list[threadIdx.x].particle_id;
    if (reference_particle_id == -1)
        return;

    cell_list[blockIdx.x].particle_list[threadIdx.x].vx += accelerations[reference_particle_id] * _TIMESTEP_DURATION;
    cell_list[blockIdx.x].particle_list[threadIdx.x].vy += accelerations[reference_particle_id + 1] * _TIMESTEP_DURATION;
    cell_list[blockIdx.x].particle_list[threadIdx.x].vz += accelerations[reference_particle_id + 2] * _TIMESTEP_DURATION;
    cell_list[blockIdx.x].particle_list[threadIdx.x].x = (cell_list[blockIdx.x].particle_list[threadIdx.x].x + cell_list[blockIdx.x].particle_list[threadIdx.x].vx * _TIMESTEP_DURATION) - (_CELL_LENGTH_X * _CELL_CUTOFF_RADIUS) * floor(cell_list[blockIdx.x].particle_list[threadIdx.x].x / (_CELL_LENGTH_X * _CELL_CUTOFF_RADIUS));
    cell_list[blockIdx.x].particle_list[threadIdx.x].y = (cell_list[blockIdx.x].particle_list[threadIdx.x].y + cell_list[blockIdx.x].particle_list[threadIdx.x].vy * _TIMESTEP_DURATION) - (_CELL_LENGTH_Y * _CELL_CUTOFF_RADIUS) * floor(cell_list[blockIdx.x].particle_list[threadIdx.x].y / (_CELL_LENGTH_Y * _CELL_CUTOFF_RADIUS));
    cell_list[blockIdx.x].particle_list[threadIdx.x].z = (cell_list[blockIdx.x].particle_list[threadIdx.x].z + cell_list[blockIdx.x].particle_list[threadIdx.x].vz * _TIMESTEP_DURATION) - (_CELL_LENGTH_Z * _CELL_CUTOFF_RADIUS) * floor(cell_list[blockIdx.x].particle_list[threadIdx.x].z / (_CELL_LENGTH_Z * _CELL_CUTOFF_RADIUS));

    accelerations[reference_particle_id] = 0;
}

// update cell lists because particles have moved
__global__ void motion_update(struct Cell *cell_list_src, struct Cell *cell_list_dst)
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
    int home_x = blockIdx.x % _CELL_LENGTH_X;
    int home_y = blockIdx.x / _CELL_LENGTH_X % _CELL_LENGTH_Y;
    int home_z = blockIdx.x / (_CELL_LENGTH_X * _CELL_LENGTH_Y) % _CELL_LENGTH_Z;

    // location of where thread is in buffer
    int free_idx = 0;

    for (int current_cell_idx = 0; current_cell_idx < _CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z; ++current_cell_idx) {
        for (int particle_idx = 0; particle_idx < _MAX_PARTICLES_PER_CELL && cell_list_src[current_cell_idx].particle_list[particle_idx].particle_id != -1; ++particle_idx) {
            struct Particle current_particle = cell_list_src[current_cell_idx].particle_list[particle_idx];
            int new_cell_x = current_particle.x / (_CELL_LENGTH_X * _CELL_CUTOFF_RADIUS);
            int new_cell_y = current_particle.y / (_CELL_LENGTH_Y * _CELL_CUTOFF_RADIUS);
            int new_cell_z = current_particle.z / (_CELL_LENGTH_Z * _CELL_CUTOFF_RADIUS);

            if (home_x == new_cell_x && home_y == new_cell_y && home_z == new_cell_z) {
                cell_list_dst[home_x + home_y * _CELL_LENGTH_X + home_z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[free_idx++] = current_particle;
            }
        }
    }
    cell_list_dst[home_x + home_y * _CELL_LENGTH_X + home_z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[free_idx].particle_id = -1;
}

// initialize cells with random particle data
void initialize_cell_list(struct Cell cellList[_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z])
{
    // initialize cell list, -1 for empty cell
    memset(cellList, -1, sizeof(struct Cell)*_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z);
    for (int i = 0; i < _NUM_PARTICLES; ++i) {
        int x = rand() % _CELL_LENGTH_X;
        int y = rand() % _CELL_LENGTH_Y;
        int z = rand() % _CELL_LENGTH_Z;
        // assign random particle data
        struct Particle particle = {
            .particle_id = i,
            .x = x * _CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * _CELL_CUTOFF_RADIUS,
            .y = y * _CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * _CELL_CUTOFF_RADIUS,
            .z = z * _CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * _CELL_CUTOFF_RADIUS,
            .vx = 0,
            .vy = 0,
            .vz = 0,
        };
        // copy particle to to cell list
        for (int j = 0; j < _MAX_PARTICLES_PER_CELL; ++j) {
            if (cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[j].particle_id == -1) {
                memcpy(&cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[j], &particle, sizeof(struct Particle));
                break;
            }
        }
    }
    for (int x = 0; x < _CELL_LENGTH_X; ++x) {
        for (int y = 0; y < _CELL_LENGTH_Y; ++y) {
            for (int z = 0; z < _CELL_LENGTH_Z; ++z) {
                int count = 0;
                while (cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].particle_id != -1) {
                    printf("%d: (%f, %f, %f)\n", cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].particle_id, cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].x , cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].y, cellList[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].z);
                    count++;
                }
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
    struct Cell cell_list[_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z];
    initialize_cell_list(cell_list);
    // device_cell_list stores an array of Cells, where each Cell contains a particle_list
    struct Cell *device_cell_list;
    // cudaMalloc initializes GPU global memory to be used as parameter for GPU kernel
    cudaMalloc(&device_cell_list, _CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z * sizeof(struct Cell) * 2);
    cudaMemcpy(device_cell_list, cell_list, _CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice);


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
    cudaMalloc(&accelerations, _MAX_PARTICLES_PER_CELL * 3 * sizeof(float));
    cudaMemset(accelerations, 0, _MAX_PARTICLES_PER_CELL * 3 * sizeof(float));


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE PARAMETERS FOR FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // defines block and thread dimensions
    // dim3 is an integer vector type most commonly used to pass the grid and block dimensions in a kernel invocation [X x Y x Z]
    // there are 2^31 blocks in x dimension while y and z have at most 65536 blocks
    dim3 numBlocksForce(_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z * 14);        // (_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z * 14) x 1 x 1
    dim3 numBlocksParticle(_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z);
    dim3 numBlocksMotion(_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z);            // (_CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z) x 1 x 1
    dim3 threadsPerBlockForce(_MAX_PARTICLES_PER_CELL);                              // _MAX_PARTICLES_PER_CELL x 1 x 1
    dim3 threadsPerBlockParticle(_MAX_PARTICLES_PER_CELL);                              // _MAX_PARTICLES_PER_CELL x 1 x 1
//    dim3 threadsPerBlockMotion(_CELL_LENGTH_X, _CELL_LENGTH_Y, _CELL_LENGTH_Z);  

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // do force evaluation and motion update for each time step
    // steps are separated to ensure threads are synchronized (that force_eval is done)
    // output of force_eval is stores in device_cell_list and accelerations

    int flag = 1;

    // address + ((flag == 1) * sizeof(cell list))

    // if flag == 0, then pass in address
    // if flag == 1, then pass in address + offset
    // flag = !flag;

    for (int t = 0; t < _TIMESTEPS; ++t) {
        if (flag) {
            force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list, accelerations);
            particle_update<<<numBlocksParticle, threadsPerBlockParticle>>>(device_cell_list, accelerations);
            motion_update<<<numBlocksMotion, 1>>>(device_cell_list, device_cell_list + sizeof(cell_list));
        } else {
            force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list + sizeof(cell_list), accelerations);
            particle_update<<<numBlocksParticle, threadsPerBlockParticle>>>(device_cell_list, accelerations);
            motion_update<<<numBlocksMotion, 1>>>(device_cell_list + sizeof(cell_list), device_cell_list);
        }
        flag = !flag;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  COPY FINAL RESULT BACK TO HOST CPU
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // if flag == 0, then results are in the second half
    // if flag == 1, then results are in the first half
    if (flag) {
        cudaMemcpy(cell_list, device_cell_list, _CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(cell_list, device_cell_list + sizeof(cell_list), _CELL_LENGTH_X * _CELL_LENGTH_Y * _CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost);
    }
    cudaFree(device_cell_list);

    for (int x = 0; x < _CELL_LENGTH_X; ++x) {
        for (int y = 0; y < _CELL_LENGTH_Y; ++y) {
            for (int z = 0; z < _CELL_LENGTH_Z; ++z) {
                int count = 0;
                while (cell_list[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].particle_id != -1) {
                    printf("%d: (%f, %f, %f)\n", cell_list[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].particle_id, cell_list[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].x , cell_list[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].y, cell_list[x + y * _CELL_LENGTH_X + z * _CELL_LENGTH_X * _CELL_LENGTH_Y].particle_list[count].z);
                    count++;
                }
            }
        }
    }

    return 0;
}
