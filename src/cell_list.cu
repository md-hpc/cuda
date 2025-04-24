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

__global__ void force_eval(const struct Cell *cell_list, float *accelerations)
{
    int home_x = blockIdx.x % CELL_LENGTH_X;
    int home_y = (blockIdx.x / CELL_LENGTH_X) % CELL_LENGTH_Y;
    int home_z = blockIdx.x / (CELL_LENGTH_Y * CELL_LENGTH_X);

    int neighbor_x;
    switch (blockIdx.y % 3) {
    case 0:
        neighbor_x = MINUS_1(home_x, CELL_LENGTH_X);
        break;
    case 1:
        neighbor_x = home_x;
        break;
    case 2:
        neighbor_x = PLUS_1(home_x, CELL_LENGTH_X);
        break;
    }

    int neighbor_y;
    switch ((blockIdx.y / 3) % 3) {
    case 0:
        neighbor_y = MINUS_1(home_y, CELL_LENGTH_Y);
        break;
    case 1:
        neighbor_y = home_y;
        break;
    case 2:
        neighbor_y = PLUS_1(home_y, CELL_LENGTH_Y);
        break;
    }

    int neighbor_z;
    switch (blockIdx.y / 9) {
    case 0:
        neighbor_z = MINUS_1(home_z, CELL_LENGTH_Z);
        break;
    case 1:
        neighbor_z = home_z;
        break;
    case 2:
        neighbor_z = PLUS_1(home_z, CELL_LENGTH_Z);
        break;
    }

    int neighbor_idx = neighbor_x + neighbor_y * CELL_LENGTH_X + neighbor_z * CELL_LENGTH_X * CELL_LENGTH_Y;

    __shared__ struct Cell neighbor_cell;
    neighbor_cell.particle_ids[threadIdx.x] = cell_list[neighbor_idx].particle_ids[threadIdx.x];
    neighbor_cell.x[threadIdx.x] = cell_list[neighbor_idx].x[threadIdx.x];
    neighbor_cell.y[threadIdx.x] = cell_list[neighbor_idx].y[threadIdx.x];
    neighbor_cell.z[threadIdx.x] = cell_list[neighbor_idx].z[threadIdx.x];

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

    if (cell_list[blockIdx.x].particle_ids[threadIdx.x] == -1)
        return;

    __syncthreads();

    float reference_x = cell_list[blockIdx.x].x[threadIdx.x];
    float reference_y = cell_list[blockIdx.x].y[threadIdx.x];
    float reference_z = cell_list[blockIdx.x].z[threadIdx.x];

    float reference_ax = 0;
    float reference_ay = 0;
    float reference_az = 0;

    // TODO: consider the situation where home = neighbor
    for (int i = 0; i < MAX_PARTICLES_PER_CELL; ++i) {
        if (neighbor_cell.particle_ids[i] == -1)
            break;

        if (neighbor_cell.particle_ids[i] == cell_list[blockIdx.x].particle_ids[threadIdx.x])
            continue;

        float diff_x = reference_x - neighbor_cell.x[i];
        float diff_y = reference_y - neighbor_cell.y[i];
        float diff_z = reference_z - neighbor_cell.z[i];

        float norm = sqrtf((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));

        float acceleration = compute_acceleration(norm) / norm;
        reference_ax += acceleration * diff_x;
        reference_ay += acceleration * diff_y;
        reference_az += acceleration * diff_z;
    }

    int accelerations_block_idx = (blockIdx.x * 27 + blockIdx.y) * MAX_PARTICLES_PER_CELL * 3;
    accelerations[accelerations_block_idx + threadIdx.x] = reference_ax;
    accelerations[accelerations_block_idx + threadIdx.x + MAX_PARTICLES_PER_CELL] = reference_ay;
    accelerations[accelerations_block_idx + threadIdx.x + (MAX_PARTICLES_PER_CELL * 2)] = reference_az;

    return;
}

__global__ void particle_update(struct Cell *cell_list, float *accelerations)
{
    if (cell_list[blockIdx.x].particle_ids[threadIdx.x] == -1)
        return;

    float reference_vx = cell_list[blockIdx.x].vx[threadIdx.x];
    float reference_vy = cell_list[blockIdx.x].vy[threadIdx.x];
    float reference_vz = cell_list[blockIdx.x].vz[threadIdx.x];
    float reference_x = cell_list[blockIdx.x].x[threadIdx.x];
    float reference_y = cell_list[blockIdx.x].y[threadIdx.x];
    float reference_z = cell_list[blockIdx.x].z[threadIdx.x];

    float ax = 0;
    float ay = 0;
    float az = 0;

    for (int i = 0; i < 27; ++i) {
        int accelerations_block_idx = (blockIdx.x * 27 + i) * MAX_PARTICLES_PER_CELL * 3;
        ax += accelerations[accelerations_block_idx + threadIdx.x];
        accelerations[accelerations_block_idx + threadIdx.x] = 0;
        ay += accelerations[accelerations_block_idx + threadIdx.x + MAX_PARTICLES_PER_CELL];
        accelerations[accelerations_block_idx + threadIdx.x + MAX_PARTICLES_PER_CELL] = 0;
        az += accelerations[accelerations_block_idx + threadIdx.x + (MAX_PARTICLES_PER_CELL * 2)];
        accelerations[accelerations_block_idx + threadIdx.x + (MAX_PARTICLES_PER_CELL * 2)] = 0;
    }

    reference_vx += ax * TIMESTEP_DURATION_FS;
    reference_vy += ay * TIMESTEP_DURATION_FS;
    reference_vz += az * TIMESTEP_DURATION_FS;

    float x = reference_x + reference_vx * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
    reference_x = x;

    float y = reference_y + reference_vy * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
    reference_y = y;

    float z = reference_z + reference_vz * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
    reference_z = z;

    cell_list[blockIdx.x].vx[threadIdx.x] = reference_vx;
    cell_list[blockIdx.x].vy[threadIdx.x] = reference_vy;
    cell_list[blockIdx.x].vz[threadIdx.x] = reference_vz;
    cell_list[blockIdx.x].x[threadIdx.x] = reference_x;
    cell_list[blockIdx.x].y[threadIdx.x] = reference_y;
    cell_list[blockIdx.x].z[threadIdx.x] = reference_z;

    return;
}

__global__ void motion_update(struct Cell *cell_list_src, struct Cell *cell_list_dst)
{
    int home_x = blockIdx.x % CELL_LENGTH_X;
    int home_y = (blockIdx.x / CELL_LENGTH_X) % CELL_LENGTH_Y;
    int home_z = blockIdx.x / (CELL_LENGTH_X * CELL_LENGTH_Y);

    __shared__ int free_idx;
    if (threadIdx.x == 0)
        free_idx = 0;
    __syncthreads();

    // can maybe make the optimization of only looking at neighboring cells on the assumption that particles don't move more than one cell in a timestep
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

    int particle_count;

    int *host_particle_ids = NULL;
    float *host_x = NULL;
    float *host_y = NULL;
    float *host_z = NULL;
    struct Cell host_cell_list[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z];

    struct Cell *device_cell_list_1;
    struct Cell *device_cell_list_2;
    float *accelerations;

    import_atoms(input_file, &host_particle_ids, &host_x, &host_y, &host_z, &particle_count);
    create_cell_list(host_particle_ids, host_x, host_y, host_z, particle_count, host_cell_list, CELL_CUTOFF_RADIUS_ANGST, CELL_LENGTH_X, CELL_LENGTH_Y, CELL_LENGTH_Z);

    GPU_PERROR(cudaMalloc(&device_cell_list_1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMemcpy(device_cell_list_1, host_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice));

    GPU_PERROR(cudaMalloc(&device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMalloc(&accelerations, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 27 * MAX_PARTICLES_PER_CELL * 3 * sizeof(float)));
    GPU_PERROR(cudaMemset(accelerations, 0, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 27 * MAX_PARTICLES_PER_CELL * 3 * sizeof(float)));

    dim3 numBlocksCalculate(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, 27);
    dim3 numBlocksUpdate(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_CELL);

#ifdef TIME_RUN
    cudaEvent_t time_start;
    cudaEvent_t time_stop;
    GPU_PERROR(cudaEventCreate(&time_start));
    GPU_PERROR(cudaEventCreate(&time_stop));

    GPU_PERROR(cudaEventRecord(time_start));
#endif

    for (int t = 0; t < TIMESTEPS; ++t) {
        if (t % 2 == 0) {
            force_eval<<<numBlocksCalculate, threadsPerBlock>>>(device_cell_list_1, accelerations);
            particle_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_1, accelerations);
            motion_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_1, device_cell_list_2);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
#endif
        } else {
            force_eval<<<numBlocksCalculate, threadsPerBlock>>>(device_cell_list_2, accelerations);
            particle_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_2, accelerations);
            motion_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_2, device_cell_list_1);
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
    printf("cell_list,%d,%f\n", particle_count, elapsed_milliseconds / 1000);

    cell_list_to_csv(host_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, output_file, "a");
#endif

    GPU_PERROR(cudaFree(device_cell_list_1));
    GPU_PERROR(cudaFree(device_cell_list_2));
    GPU_PERROR(cudaFree(accelerations));

    return 0;
}
