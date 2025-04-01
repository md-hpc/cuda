extern "C" {
#include "pdb_importer.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

//#define EPSILON (1.65e-9)                       // ng * m^2 / s^2
#define EPSILON (1.65e11)                        // ng * A^2 / s^2
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (0.034f)                           // A
#define PLUS_1(dimension, length) ((dimension != length - 1) * (dimension + 1))
#define MINUS_1(dimension, length) ((dimension == 0) * length + dimension - 1)
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);

__device__ float compute_acceleration(float r_angstrom) {
        // in A / s^2
        float temp = SIGMA / r_angstrom;
        temp = temp * temp;
        temp = temp * temp * temp;
        float acceleration = 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
        //float force = 4 * EPSILON * (12 * pow(SIGMA, 12.0f) / pow(r, 13.0f) - 6 * pow(SIGMA, 6.0f) / pow(r, 7.0f)) / ARGON_MASS;

        return (acceleration < LJMAX) * LJMAX + !(acceleration < LJMAX) * acceleration;
}

__global__ void force_eval(struct Cell *cell_list, float *accelerations)
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
    neighbor_cell.particle_id[threadIdx.x] = cell_list[neighbor_idx].particle_id[threadIdx.x];
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

    if (cell_list[blockIdx.x].particle_id[threadIdx.x] == -1)
        return;

    __syncthreads();

    float reference_x = cell_list[blockIdx.x].x[threadIdx.x];
    float reference_y = cell_list[blockIdx.x].y[threadIdx.x];
    float reference_z = cell_list[blockIdx.x].z[threadIdx.x];

    float reference_particle_ax = 0;
    float reference_particle_ay = 0;
    float reference_particle_az = 0;

    // TODO: consider the situation where home = neighbor
    for (int i = 0; i < MAX_PARTICLES_PER_CELL; ++i) {
        if (neighbor_cell.particle_id[i] == -1)
            break;

        if (neighbor_cell.particle_id[i] == cell_list[blockIdx.x].particle_ids[threadIdx.x])
            continue;

        float diff_x = reference_x - neighbor_cell.x[i];
        float diff_y = reference_y - neighbor_cell.y[i];
        float diff_z = reference_z - neighbor_cell.z[i];

        float norm = sqrtf((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));

        float acceleration = compute_acceleration(norm);
        reference_particle_ax += acceleration * diff_x / norm;
        reference_particle_ay += acceleration * diff_y / norm;
        reference_particle_az += acceleration * diff_z / norm;
    }

    int accelerations_block_idx = ((blockIdx.x * 27 + blockIdx.y) * MAX_PARTICLES_PER_CELL + threadIdx.x) * 3;
    accelerations[accelerations_block_idx] = reference_particle_ax;
    accelerations[accelerations_block_idx + 1] = reference_particle_ay;
    accelerations[accelerations_block_idx + 2] = reference_particle_az;

    return;
}

__global__ void particle_update(struct Cell *cell_list, float *accelerations)
{
    struct Particle reference_particle = cell_list[blockIdx.x].particle_list[threadIdx.x];
    if (reference_particle.particle_id == -1)
        return;

    float ax = 0;
    float ay = 0;
    float az = 0;

    for (int i = 0; i < 27; ++i) {
        int accelerations_block_idx = ((blockIdx.x * 27 + i) * MAX_PARTICLES_PER_CELL + threadIdx.x) * 3;
        ax += accelerations[accelerations_block_idx];
        ay += accelerations[accelerations_block_idx + 1];
        az += accelerations[accelerations_block_idx + 2];
    }

    reference_particle.vx += ax * TIMESTEP_DURATION_FS;
    reference_particle.vy += ay * TIMESTEP_DURATION_FS;
    reference_particle.vz += az * TIMESTEP_DURATION_FS;

    float x = reference_particle.x + reference_particle.vx * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
    reference_particle.x = x;

    float y = reference_particle.y + reference_particle.vy * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
    reference_particle.y = y;

    float z = reference_particle.z + reference_particle.vz * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
    reference_particle.z = z;

    cell_list[blockIdx.x].particle_list[threadIdx.x] = reference_particle;

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
        struct Particle current_particle = cell_list_src[current_cell_idx].particle_list[threadIdx.x];
        if (current_particle.particle_id == -1)
            continue;

        int new_cell_x = current_particle.x / CELL_CUTOFF_RADIUS_ANGST;
        int new_cell_y = current_particle.y / CELL_CUTOFF_RADIUS_ANGST;
        int new_cell_z = current_particle.z / CELL_CUTOFF_RADIUS_ANGST;

        if (home_x == new_cell_x && home_y == new_cell_y && home_z == new_cell_z) {
            int idx = atomicAdd(&free_idx, 1);
            cell_list_dst[blockIdx.x].particle_list[idx] = current_particle;
        }
    }

    __syncthreads();

    if (threadIdx.x >= MAX_PARTICLES_PER_CELL - free_idx)
        cell_list_dst[blockIdx.x].particle_list[threadIdx.x].particle_id = -1;

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

    int particle_count;

    int *host_particle_ids = NULL;
    float *host_x = NULL;
    float *host_y = NULL;
    float *host_z = NULL;
    struct Cell host_cell_list[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z];

    struct Cell *device_cell_list_1;
    struct Cell *device_cell_list_1;
    float *accelerations;

    import_atoms(input_file, &host_particle_ids, &host_x, &host_y, &host_z, &particle_count);
    create_cell_list(host_particle_ids, host_x, host_y, host_z, particle_count, host_cell_list, CELL_CUTOFF_RADIUS_ANGST);

    GPU_PERROR(cudaMalloc(&device_cell_list_1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMalloc(&device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMalloc(&accelerations, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 27 * MAX_PARTICLES_PER_CELL * 3 * sizeof(float)));
    GPU_PERROR(cudaMemcpy(device_cell_list_1, host_cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice));

    dim3 numBlocksCalculate(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, 27);
    dim3 numBlocksUpdate(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_CELL);

#ifdef SIMULATE
    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
#endif

#ifdef TIME_RUN
    struct timespec time_start;
    struct timespec time_stop;
    clock_gettime(CLOCK_REALTIME, &time_start);
#endif

    int t;
    for (t = 0; t < TIMESTEPS; ++t) {
        GPU_PERROR(cudaMemset(accelerations, 0, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 27 * MAX_PARTICLES_PER_CELL * 3 * sizeof(float)));
        if (t & 1 == 0) {
            force_eval<<<numBlocksCalculate, threadsPerBlock>>>(device_cell_list_1, accelerations);
            particle_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_1, accelerations);
            motion_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_1, device_cell_list2);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_2, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
#endif
        } else {
            force_eval<<<numBlocksCalculate, threadsPerBlock>>>(device_cell_list_2, accelerations);
            particle_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_2, accelerations);
            motion_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list_2, device_cell_list1);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
#endif
        }
#ifdef SIMULATE
        for (int x = 0; x < CELL_LENGTH_X; ++x) {
            for (int y = 0; y < CELL_LENGTH_Y; ++y) {
                for (int z = 0; z < CELL_LENGTH_Z; ++z) {
                    int count = 0;
                    struct Cell current_cell = cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y];
                    while (count < MAX_PARTICLES_PER_CELL && current_cell.particle_id[count] != -1) {
                        fprintf(out, "%d,%d,%f,%f,%f\n", x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y,
                                                        current_cell.particle_id[count],
                                                        current_cell.x[count],
                                                        current_cell.y[count],
                                                        current_cell.z[count]);
                        count++;
                    }
                }
            }
        }
        fprintf(out, "\n");
#endif
    }

#ifdef TIME_RUN
    clock_gettime(CLOCK_REALTIME, &time_stop);

    struct timespec temp;
    temp.tv_sec = time_stop.tv_sec - time_start.tv_sec;
    temp.tv_nsec = time_stop.tv_nsec - time_start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }

    printf("cell_list,%f\n", ((double) temp.tv_sec) + (((double) temp.tv_nsec) * 1e-9));

    if (t & 1) {
        GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(host_cell_list, device_cell_list_1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    }

    FILE *out = fopen(output_file, "w");
    fprintf(out, "cell_idx,particle_id,x,y,z\n");

    for (int x = 0; x < CELL_LENGTH_X; ++x) {
        for (int y = 0; y < CELL_LENGTH_Y; ++y) {
            for (int z = 0; z < CELL_LENGTH_Z; ++z) {
                int count = 0;
                struct Cell current_cell = cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y];
                while (count < MAX_PARTICLES_PER_CELL && current_cell.particle_id[count] != -1) {
                    fprintf(out, "%d,%d,%f,%f,%f\n", x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y,
                                                     current_cell.particle_id[count],
                                                     current_cell.x[count],
                                                     current_cell.y[count],
                                                     current_cell.z[count]);
                    count++;
                }
            }
        }
    }
#endif

    GPU_PERROR(cudaFree(device_cell_list_1));
    GPU_PERROR(cudaFree(device_cell_list_2));
}
