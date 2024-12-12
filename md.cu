#include <stdio.h>
#include <stdlib.h>
#include <curand.h>


// minimum max shared memory size per SM across all architectures is 64K
// minimum max resident block per SM across all architectures is 16
// so worst case, each block will have max 4K shared memory

// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define NUM_PARTICLES 3
#define MAX_PARTICLES_PER_CELL 128

#define CELL_CUTOFF_RADIUS_ANGST 100
#define CELL_LENGTH_X 3
#define CELL_LENGTH_Y 3
#define CELL_LENGTH_Z 3

#define TIMESTEPS 1
#define TIMESTEP_DURATION_FS 1                            
#define EPSILON 1.0f
#define SIGMA 1.0f
#define LJMIN (-4.0f * 24.0f * EPSILON / SIGMA * (powf(7.0f / 26.0f, 7.0f / 6.0f) - 2.0f * powf(7.0f / 26.0f, 13.0f / 6.0f)))

#define PLUS_1(dimension, length) ((dimension != length - 1) * (dimension + 1))
#define MINUS_1(dimension, length) ((dimension == 0) * length + dimension - 1)
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);
	


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
    struct Particle particle_list[MAX_PARTICLES_PER_CELL];
};

// LJ force computation
__device__ float compute_acceleration(float r1, float r2) {
    float r = fabsf(r1 - r2);
    float force = 4 * EPSILON * (6 * powf(SIGMA, 6.0f) / powf(r, 7.0f) - 12 * powf(SIGMA, 12.0f) / powf(r, 13.0f));

    return (force < LJMIN) * LJMIN + !(force < LJMIN) * force;
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

    // find hcell coordinate based off of block index x
    int home_x = blockIdx.x % CELL_LENGTH_X;
    int home_y = blockIdx.x / CELL_LENGTH_X % CELL_LENGTH_Y;
    int home_z = blockIdx.x / (CELL_LENGTH_Y * CELL_LENGTH_X) % CELL_LENGTH_Z;

    // find ncell coordinate based off of block index y
    int neighbor_x;
    if (blockIdx.y < 9) {
        neighbor_x = PLUS_1(home_x, CELL_LENGTH_X);
    } else {
        neighbor_x = home_x;
    }

    int neighbor_y;
    if (blockIdx.y < 3) {
        neighbor_y = MINUS_1(home_y, CELL_LENGTH_Y);
    } else if (blockIdx.y >= 3 && blockIdx.y <= 5 || blockIdx.y > 11) {
        neighbor_y = home_y;
    } else {
        neighbor_y = PLUS_1(home_y, CELL_LENGTH_Y);
    }

    int neighbor_z;
    if (blockIdx.y % 3 == 0) {
        neighbor_z = PLUS_1(home_z, CELL_LENGTH_Z);
    } else if (blockIdx.y % 3 == 1) {
        neighbor_z = home_z;
    } else {
        neighbor_z = MINUS_1(home_z, CELL_LENGTH_Z);
    }

    int neighbor_is_home = home_x == neighbor_x && home_y == neighbor_y && home_z == neighbor_z;

    // define and assign shared memory
    __shared__ struct Cell neighbor_cell;
    int neighbor_idx = neighbor_x + neighbor_y * CELL_LENGTH_X + neighbor_z * CELL_LENGTH_X * CELL_LENGTH_Y;
    neighbor_cell.particle_list[threadIdx.x].particle_id = cell_list[neighbor_idx].particle_list[threadIdx.x].particle_id;
    neighbor_cell.particle_list[threadIdx.x].x = cell_list[neighbor_idx].particle_list[threadIdx.x].x;
    neighbor_cell.particle_list[threadIdx.x].y = cell_list[neighbor_idx].particle_list[threadIdx.x].y;
    neighbor_cell.particle_list[threadIdx.x].z = cell_list[neighbor_idx].particle_list[threadIdx.x].z;

    // for periodic boundary condition
    if (!neighbor_is_home) {
        if (home_x - neighbor_x == CELL_LENGTH_X - 1)
            neighbor_cell.particle_list[threadIdx.x].x += (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
        else if (neighbor_x - home_x == CELL_LENGTH_X - 1)
            neighbor_cell.particle_list[threadIdx.x].x -= (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
        if (home_y - neighbor_y == CELL_LENGTH_Y - 1)
            neighbor_cell.particle_list[threadIdx.x].y += (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
        else if (neighbor_y - home_y == CELL_LENGTH_Y - 1)
            neighbor_cell.particle_list[threadIdx.x].y -= (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
        if (home_z - neighbor_z == CELL_LENGTH_Z - 1)
            neighbor_cell.particle_list[threadIdx.x].z += (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
        else if (neighbor_z - home_z == CELL_LENGTH_Z - 1)
            neighbor_cell.particle_list[threadIdx.x].z -= (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
    }

    // synchronizes threads within a block (all threads must complete tasks)
    __syncthreads();

    int home_idx = home_x + home_y * CELL_LENGTH_X + home_z * CELL_LENGTH_X * CELL_LENGTH_Y;
    int reference_particle_id = cell_list[home_idx].particle_list[threadIdx.x].particle_id;
    // if particle exists loop through every particle in ncell particle list
    if (reference_particle_id != -1) {
        int reference_particle_x = cell_list[home_idx].particle_list[threadIdx.x].x;
        int reference_particle_y = cell_list[home_idx].particle_list[threadIdx.x].y;
        int reference_particle_z = cell_list[home_idx].particle_list[threadIdx.x].z;

        float reference_particle_ax = 0;
        float reference_particle_ay = 0;
        float reference_particle_az = 0;

        for (int i = 0; i < MAX_PARTICLES_PER_CELL; ++i) {
            if (neighbor_cell.particle_list[i].particle_id == -1)
                break;

            if (neighbor_is_home && !(reference_particle_x < neighbor_cell.particle_list[i].x))
                continue;

            float ax = compute_acceleration(reference_particle_x, neighbor_cell.particle_list[i].x);
            float ay = compute_acceleration(reference_particle_y, neighbor_cell.particle_list[i].y);
            float az = compute_acceleration(reference_particle_z, neighbor_cell.particle_list[i].z);

            reference_particle_ax += ax;
            reference_particle_ay += ay;
            reference_particle_az += az;

            int neighbor_particle_id = neighbor_cell.particle_list[i].particle_id;
            atomicAdd(&accelerations[neighbor_particle_id * 3], -ax);
            atomicAdd(&accelerations[neighbor_particle_id * 3 + 1], -ay);
            atomicAdd(&accelerations[neighbor_particle_id * 3 + 2], -az);
        }

        atomicAdd(&accelerations[reference_particle_id * 3], reference_particle_ax);
        atomicAdd(&accelerations[reference_particle_id * 3 + 1], reference_particle_ay);
        atomicAdd(&accelerations[reference_particle_id * 3 + 2], reference_particle_az);
    }
}

__global__ void particle_update(struct Cell *cell_list, float *accelerations)
{
    // 1 block -> 1 cell
    // 1 thread -> 1 particle

    int reference_particle_id = cell_list[blockIdx.x].particle_list[threadIdx.x].particle_id;
    if (reference_particle_id == -1)
        return;

    cell_list[blockIdx.x].particle_list[threadIdx.x].vx += accelerations[reference_particle_id * 3] * TIMESTEP_DURATION_FS;
    cell_list[blockIdx.x].particle_list[threadIdx.x].vy += accelerations[reference_particle_id * 3 + 1] * TIMESTEP_DURATION_FS;
    cell_list[blockIdx.x].particle_list[threadIdx.x].vz += accelerations[reference_particle_id * 3 + 2] * TIMESTEP_DURATION_FS;

    float x = cell_list[blockIdx.x].particle_list[threadIdx.x].x + cell_list[blockIdx.x].particle_list[threadIdx.x].vx * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
    cell_list[blockIdx.x].particle_list[threadIdx.x].x = x;

    float y = cell_list[blockIdx.x].particle_list[threadIdx.x].y + cell_list[blockIdx.x].particle_list[threadIdx.x].vy * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
    cell_list[blockIdx.x].particle_list[threadIdx.x].y = y;

    float z = cell_list[blockIdx.x].particle_list[threadIdx.x].z + cell_list[blockIdx.x].particle_list[threadIdx.x].vz * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
    cell_list[blockIdx.x].particle_list[threadIdx.x].z = z;

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
    int home_x = blockIdx.x % CELL_LENGTH_X;
    int home_y = blockIdx.x / CELL_LENGTH_X % CELL_LENGTH_Y;
    int home_z = blockIdx.x / (CELL_LENGTH_X * CELL_LENGTH_Y) % CELL_LENGTH_Z;

    // location of where thread is in buffer
    __shared__ int free_idx;
    if (threadIdx.x == 0)
        free_idx = 0;
    __syncthreads();

    for (int current_cell_idx = 0; current_cell_idx < CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z; ++current_cell_idx) {
        struct Particle current_particle = cell_list_src[current_cell_idx].particle_list[threadIdx.x];
        if (current_particle.particle_id == -1)
            continue;

        int new_cell_x = current_particle.x / (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
        int new_cell_y = current_particle.y / (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
        int new_cell_z = current_particle.z / (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);

        if (home_x == new_cell_x && home_y == new_cell_y && home_z == new_cell_z) {
            int idx = atomicAdd(&free_idx, 1);
            cell_list_dst[blockIdx.x].particle_list[idx] = current_particle;
        }
    }

    if (threadIdx.x == 0)
        cell_list_dst[blockIdx.x].particle_list[free_idx].particle_id = -1;
}

// initialize cells with random particle data
void initialize_cell_list(struct Cell cellList[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z])
{
    // initialize cell list, -1 for empty cell
    memset(cellList, -1, sizeof(struct Cell)*CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int x = rand() % CELL_LENGTH_X;
        int y = rand() % CELL_LENGTH_Y;
        int z = rand() % CELL_LENGTH_Z;
        // assign random particle data
        struct Particle particle = {
            .particle_id = i,
            .x = x * CELL_CUTOFF_RADIUS_ANGST + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS_ANGST,
            .y = y * CELL_CUTOFF_RADIUS_ANGST + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS_ANGST,
            .z = z * CELL_CUTOFF_RADIUS_ANGST + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS_ANGST,
            .vx = 0,
            .vy = 0,
            .vz = 0,
        };
        // copy particle to to cell list
        for (int j = 0; j < MAX_PARTICLES_PER_CELL; ++j) {
            if (cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[j].particle_id == -1) {
                memcpy(&cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[j], &particle, sizeof(struct Particle));
                break;
            }
        }
    }
    for (int x = 0; x < CELL_LENGTH_X; ++x) {
        for (int y = 0; y < CELL_LENGTH_Y; ++y) {
            for (int z = 0; z < CELL_LENGTH_Z; ++z) {
                int count = 0;
                while (cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].particle_id != -1) {
                    printf("%d: (%f, %f, %f)\n", cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].particle_id, cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].x , cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].y, cellList[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].z);
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
    struct Cell cell_list[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z];
    memset(cell_list, -1, sizeof(struct Cell)*CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);
    //initialize_cell_list(cell_list);
    cell_list[0].particle_list[0].particle_id = 0;
    cell_list[0].particle_list[0].x = 0;
    cell_list[0].particle_list[0].y = 0;
    cell_list[0].particle_list[0].z = 0;
    cell_list[0].particle_list[0].vx = 0;
    cell_list[0].particle_list[0].vy = 0;
    cell_list[0].particle_list[0].vz = 0;

    cell_list[0].particle_list[1].particle_id = 1;
    cell_list[0].particle_list[1].x = 1;
    cell_list[0].particle_list[1].y = 1;
    cell_list[0].particle_list[1].z = 1;
    cell_list[0].particle_list[1].vx = 0;
    cell_list[0].particle_list[1].vy = 0;
    cell_list[0].particle_list[1].vz = 0;
    // device_cell_list stores an array of Cells, where each Cell contains a particle_list
    struct Cell *device_cell_list1;
    struct Cell *device_cell_list2;
    // cudaMalloc initializes GPU global memory to be used as parameter for GPU kernel
    GPU_PERROR(cudaMalloc(&device_cell_list1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMalloc(&device_cell_list2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMemcpy(device_cell_list1, cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice));


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
    GPU_PERROR(cudaMalloc(&accelerations, NUM_PARTICLES * 3 * sizeof(float)));
    GPU_PERROR(cudaMemset(accelerations, 0, NUM_PARTICLES * 3 * sizeof(float)));


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE PARAMETERS FOR FORCE COMPUTATION AND MOTION UPDATE
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // defines block and thread dimensions
    // dim3 is an integer vector type most commonly used to pass the grid and block dimensions in a kernel invocation [X x Y x Z]
    // there are 2^31 blocks in x dimension while y and z have at most 65536 blocks
    dim3 numBlocksForce(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, 14);        // (CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 14) x 1 x 1
    dim3 numBlocksParticle(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);
    dim3 numBlocksMotion(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);            // (CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z) x 1 x 1
    dim3 threadsPerBlockForce(MAX_PARTICLES_PER_CELL);                              // MAX_PARTICLES_PER_CELL x 1 x 1
    dim3 threadsPerBlockParticle(MAX_PARTICLES_PER_CELL);                              // MAX_PARTICLES_PER_CELL x 1 x 1
//    dim3 threadsPerBlockMotion(CELL_LENGTH_X, CELL_LENGTH_Y, CELL_LENGTH_Z);  

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

    for (int t = 0; t < TIMESTEPS; ++t) {
        if (flag) {
            force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list1, accelerations);
            particle_update<<<numBlocksParticle, threadsPerBlockParticle>>>(device_cell_list1, accelerations);
            motion_update<<<numBlocksMotion, 1>>>(device_cell_list1, device_cell_list2);
        } else {
            force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list2, accelerations);
            particle_update<<<numBlocksParticle, threadsPerBlockParticle>>>(device_cell_list2, accelerations);
            motion_update<<<numBlocksMotion, 1>>>(device_cell_list2, device_cell_list1);
        }
        flag = !flag;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  COPY FINAL RESULT BACK TO HOST CPU
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // if flag == 0, then results are in the second half
    // if flag == 1, then results are in the first half
    if (flag) {
        GPU_PERROR(cudaMemcpy(cell_list, device_cell_list1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(cell_list, device_cell_list2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    }
    GPU_PERROR(cudaFree(device_cell_list1));
    GPU_PERROR(cudaFree(device_cell_list2));

    for (int x = 0; x < CELL_LENGTH_X; ++x) {
        for (int y = 0; y < CELL_LENGTH_Y; ++y) {
            for (int z = 0; z < CELL_LENGTH_Z; ++z) {
                int count = 0;
                while (cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].particle_id != -1) {
                    printf("%d: (%f, %f, %f)\n", cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].particle_id, cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].x , cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].y, cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y].particle_list[count].z);
                    count++;
                }
            }
        }
    }

    return 0;
}
