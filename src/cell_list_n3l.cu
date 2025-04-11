extern "C" {
#include "pdb_importer.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

#if defined(SIMULATE) && defined(TIME_RUN)
#error Cannot compile with both SIMULATE and TIME_RUN flags
#endif

#if !defined(SIMULATE) && !defined(TIME_RUN)
#error Cannot compile without neither SIMULATE nor TIME_RUN flags
#endif

// minimum max shared memory size per SM across all architectures is 64K
// minimum max resident block per SM across all architectures is 16
// so worst case, each block will have max 4K shared memory

// use profiler to identify optimal size ie. CUDA occupancy API, nvvp
#define MAX_PARTICLES_PER_CELL 128

//#define EPSILON (1.65e-9)                       // ng * m^2 / s^2
#define EPSILON (1.65e11)                       // ng * A^2 / s^2 originally (1.65e-9)
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (3.4f)                            // Angstrom
#define PLUS_1(dimension, length) ((dimension != length - 1) * (dimension + 1))
#define MINUS_1(dimension, length) ((dimension == 0) * length + dimension - 1)
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);

#define R_MIN (3.13796173693f)                  // r(-4 * V(2^(1/6) * EPSILON))
#define LJMAX_ACCELERATION (6.8829688151e25)    // A(R_MIN) in Angstrom / s^2

__device__ float compute_acceleration(float r_angstrom) {
        if (r_angstrom < R_MIN)
            return LJMAX_ACCELERATION;

        // in A / s^2
        float temp = SIGMA / r_angstrom;
        temp = temp * temp; // ^2
        temp = temp * temp * temp; // ^ 6

        return 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
}

// the meat:
__global__ void force_eval(const struct Cell *cell_list, float *accelerations)
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

    int neighbor_idx = neighbor_x + neighbor_y * CELL_LENGTH_X + neighbor_z * CELL_LENGTH_X * CELL_LENGTH_Y;

    // define and assign shared memory
    __shared__ struct Cell neighbor_cell;
    neighbor_cell.particle_ids[threadIdx.x] = cell_list[neighbor_idx].particle_ids[threadIdx.x];
    neighbor_cell.x[threadIdx.x] = cell_list[neighbor_idx].x[threadIdx.x];
    neighbor_cell.y[threadIdx.x] = cell_list[neighbor_idx].y[threadIdx.x];
    neighbor_cell.z[threadIdx.x] = cell_list[neighbor_idx].z[threadIdx.x];

    // for periodic boundary condition
    if (blockIdx.x != neighbor_idx) {
        if (home_x - neighbor_x == CELL_LENGTH_X - 1)
            neighbor_cell.x[threadIdx.x] += (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
        else if (neighbor_x - home_x == CELL_LENGTH_X - 1)
            neighbor_cell.x[threadIdx.x] -= (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
        if (home_y - neighbor_y == CELL_LENGTH_Y - 1)
            neighbor_cell.y[threadIdx.x] += (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
        else if (neighbor_y - home_y == CELL_LENGTH_Y - 1)
            neighbor_cell.y[threadIdx.x] -= (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
        if (home_z - neighbor_z == CELL_LENGTH_Z - 1)
            neighbor_cell.z[threadIdx.x] += (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
        else if (neighbor_z - home_z == CELL_LENGTH_Z - 1)
            neighbor_cell.z[threadIdx.x] -= (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
    }

    // synchronizes threads within a block (all threads must complete tasks)
    __syncthreads();

    int reference_id = cell_list[blockIdx.x].particle_ids[threadIdx.x];
    // if particle exists loop through every particle in ncell particle list
    if (reference_id != -1) {
        float reference_x = cell_list[blockIdx.x].x[threadIdx.x];
        float reference_y = cell_list[blockIdx.x].y[threadIdx.x];
        float reference_z = cell_list[blockIdx.x].z[threadIdx.x];

        float reference_ax = 0;
        float reference_ay = 0;
        float reference_az = 0;

        for (int i = 0; i < MAX_PARTICLES_PER_CELL; ++i) {
            int neighbor_id = neighbor_cell.particle_ids[i];
            if (neighbor_id == -1)
                break;

            if (neighbor_idx == blockIdx.x && !(reference_id < neighbor_id))
                continue;

            float diff_x = reference_x - neighbor_cell.x[i];
            float diff_y = reference_y - neighbor_cell.y[i];
            float diff_z = reference_z - neighbor_cell.z[i];

            float norm = sqrtf((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));

            float acceleration = compute_acceleration(norm) / norm;
            float ax = acceleration * diff_x;
            float ay = acceleration * diff_y;
            float az = acceleration * diff_z;

            reference_ax += ax;
            reference_ay += ay;
            reference_az += az;

            atomicAdd(&accelerations[neighbor_id * 3], -ax);
            atomicAdd(&accelerations[neighbor_id * 3 + 1], -ay);
            atomicAdd(&accelerations[neighbor_id * 3 + 2], -az);
        }

        atomicAdd(&accelerations[reference_id * 3], reference_ax);
        atomicAdd(&accelerations[reference_id * 3 + 1], reference_ay);
        atomicAdd(&accelerations[reference_id * 3 + 2], reference_az);
    }
}

__global__ void particle_update(struct Cell *cell_list, float *accelerations)
{
    // 1 block -> 1 cell
    // 1 thread -> 1 particle

    int reference_id = cell_list[blockIdx.x].particle_ids[threadIdx.x];
    if (reference_id == -1)
        return;

    cell_list[blockIdx.x].vx[threadIdx.x] += accelerations[reference_id * 3] * TIMESTEP_DURATION_FS;
    cell_list[blockIdx.x].vy[threadIdx.x] += accelerations[reference_id * 3 + 1] * TIMESTEP_DURATION_FS;
    cell_list[blockIdx.x].vz[threadIdx.x] += accelerations[reference_id * 3 + 2] * TIMESTEP_DURATION_FS;

    float x = cell_list[blockIdx.x].x[threadIdx.x] + cell_list[blockIdx.x].vx[threadIdx.x] * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
    cell_list[blockIdx.x].x[threadIdx.x] = x;

    float y = cell_list[blockIdx.x].y[threadIdx.x] + cell_list[blockIdx.x].vy[threadIdx.x] * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
    cell_list[blockIdx.x].y[threadIdx.x] = y;

    float z = cell_list[blockIdx.x].z[threadIdx.x] + cell_list[blockIdx.x].vz[threadIdx.x] * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
    cell_list[blockIdx.x].z[threadIdx.x] = z;

    accelerations[reference_id] = 0;
}

// update cell lists because particles have moved
__global__ void motion_update(struct Cell *cell_list_src, struct Cell *cell_list_dst)
{
    /*
        1 block per cell
        right now 1 thread per particle in a block
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
        int current_particle_id = cell_list_src[current_cell_idx].particle_ids[threadIdx.x];
        if (current_particle_id == -1)
            continue;

        float current_particle_x = cell_list_src[current_cell_idx].x[threadIdx.x];
        float current_particle_y = cell_list_src[current_cell_idx].y[threadIdx.x];
        float current_particle_z = cell_list_src[current_cell_idx].z[threadIdx.x];

        int new_cell_x = current_particle_x / CELL_CUTOFF_RADIUS_ANGST;
        int new_cell_y = current_particle_y / CELL_CUTOFF_RADIUS_ANGST;
        int new_cell_z = current_particle_z / CELL_CUTOFF_RADIUS_ANGST;

        if (home_x == new_cell_x && home_y == new_cell_y && home_z == new_cell_z) {
            int idx = atomicAdd(&free_idx, 1);
            cell_list_dst[blockIdx.x].particle_ids[idx] = current_particle_id;
            cell_list_dst[blockIdx.x].x[idx] = current_particle_x;
            cell_list_dst[blockIdx.x].y[idx] = current_particle_y;
            cell_list_dst[blockIdx.x].z[idx] = current_particle_z;
            cell_list_dst[blockIdx.x].vx[idx] = cell_list_src[current_cell_idx].vx[threadIdx.x];
            cell_list_dst[blockIdx.x].vy[idx] = cell_list_src[current_cell_idx].vy[threadIdx.x];
            cell_list_dst[blockIdx.x].vz[idx] = cell_list_src[current_cell_idx].vz[threadIdx.x];
        }
    }

    __syncthreads();

    if (threadIdx.x >= free_idx)
        cell_list_dst[blockIdx.x].particle_ids[threadIdx.x] = -1;

    return;
}


int main(int argc, char **argv) 
{
    if (argc != 3) {
	    printf("Usage: ./cell_list <input_file> <output_file>\n");
	    return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
    fclose(out);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // INITIALIZE CELL LIST WITH PARTICLE DATA
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // initialize (or import) particle data for simulation
    int particle_count;

    int *host_particle_ids;
    float *host_x;
    float *host_y;
    float *host_z;
    struct Cell host_cell_list[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z];

    // device_cell_list stores an array of Cells, where each Cell contains a particle_list
    struct Cell *device_cell_list_1;
    struct Cell *device_cell_list_2;
    float *accelerations;

    // import particles from PDB file
    import_atoms(input_file, &host_particle_ids, &host_x, &host_y, &host_z, &particle_count);
    create_cell_list(host_particle_ids, host_x, host_y, host_z, particle_count, host_cell_list, CELL_CUTOFF_RADIUS_ANGST, CELL_LENGTH_X, CELL_LENGTH_Y, CELL_LENGTH_Z);

    // cudaMalloc initializes GPU global memory to be used as parameter for GPU kernel
    GPU_PERROR(cudaMalloc(&device_cell_list_1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMemcpy(device_cell_list_1, host_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMalloc(&device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));

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
    GPU_PERROR(cudaMalloc(&accelerations, particle_count * 3 * sizeof(float)));
    GPU_PERROR(cudaMemset(accelerations, 0, particle_count * 3 * sizeof(float)));

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

#ifdef TIME_RUN
    cudaEvent_t time_start;
    cudaEvent_t time_stop;
    GPU_PERROR(cudaEventCreate(&time_start));
    GPU_PERROR(cudaEventCreate(&time_stop));

    GPU_PERROR(cudaEventRecord(time_start));
#endif    

    for (int t = 0; t < TIMESTEPS; ++t) {
        if (t % 2 == 0) {
            force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list_1, accelerations);
            particle_update<<<numBlocksParticle, threadsPerBlockParticle>>>(device_cell_list_1, accelerations);
            motion_update<<<numBlocksMotion, MAX_PARTICLES_PER_CELL>>>(device_cell_list_1, device_cell_list_2);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
#endif
        } else {
            force_eval<<<numBlocksForce, threadsPerBlockForce>>>(device_cell_list_2, accelerations);
            particle_update<<<numBlocksParticle, threadsPerBlockParticle>>>(device_cell_list_2, accelerations);
            motion_update<<<numBlocksMotion, MAX_PARTICLES_PER_CELL>>>(device_cell_list_2, device_cell_list_1);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
#endif
        }
#ifdef SIMULATE
        cell_list_to_csv(host_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, output_file, "a");
#endif
    }

#ifdef TIME_RUN
    GPU_PERROR(cudaEventRecord(time_stop));

    if (TIMESTEPS % 2 == 1) {
        GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    }

    GPU_PERROR(cudaEventSynchronize(time_stop));
    float elapsed_milliseconds = 0;
    GPU_PERROR(cudaEventElapsedTime(&elapsed_milliseconds, time_start, time_stop));
    printf("cell_list_n3l,%d,%f\n", particle_count, elapsed_milliseconds / 1000);

    cell_list_to_csv(host_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, output_file, "a");
#endif

    GPU_PERROR(cudaFree(device_cell_list_1));
    GPU_PERROR(cudaFree(device_cell_list_2));
    GPU_PERROR(cudaFree(accelerations));

    return 0;
}
