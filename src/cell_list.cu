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
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);

__device__ float compute_acceleration(float r_angstrom) {
        // in A / s^2
        float temp = (SIGMA / r_angstrom) * (SIGMA / r_angstrom) * (SIGMA / r_angstrom) * (SIGMA / r_angstrom) * (SIGMA / r_angstrom) * (SIGMA / r_angstrom);
        float acceleration = 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
        //float force = 4 * EPSILON * (12 * pow(SIGMA, 12.0f) / pow(r, 13.0f) - 6 * pow(SIGMA, 6.0f) / pow(r, 7.0f)) / ARGON_MASS;

        //return (force < LJMIN) * LJMIN + !(force < LJMIN) * force;
        return acceleration;
}

__global__ force_eval(struct Cell *cell_list, float *accelerations)
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
    neighbor_cell.particle_list[threadIdx.x].particle_id = cell_list[neighbor_idx].particle_list[threadIdx.x].particle_id;
    neighbor_cell.particle_list[threadIdx.x].x = cell_list[neighbor_idx].particle_list[threadIdx.x].x;
    neighbor_cell.particle_list[threadIdx.x].y = cell_list[neighbor_idx].particle_list[threadIdx.x].y;
    neighbor_cell.particle_list[threadIdx.x].z = cell_list[neighbor_idx].particle_list[threadIdx.x].z;

    if (blockIdx.x != neighbor_idx) {
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

    if (cell_list[blockIdx.x].particle_list[threadIdx.x].particle_id == -1)
        return;

    __syncthreads();

    float reference_particle_x = cell_list[blockIdx.x].particle_list[threadIdx.x].x;
    float reference_particle_y = cell_list[blockIdx.x].particle_list[threadIdx.x].y;
    float reference_particle_z = cell_list[blockIdx.x].particle_list[threadIdx.x].z;

    float reference_particle_ax = 0;
    float reference_particle_ay = 0;
    float reference_particle_az = 0;

    // TODO: consider the situation where home = neighbor
    for (int i = 0; i < MAX_PARTICLES_PER_CELL; ++i) {
        if (neighbor_cell.particle_list[i].particle_id == -1)
            break;

        float neighbor_particle_x = neighbor_cell.particle_list[i].x;
        float neighbor_particle_y = neighbor_cell.particle_list[i].y;
        float neighbor_particle_z = neighbor_cell.particle_list[i].z;

        float norm = sqrtf(
            (reference_particle_x - neighbor_particle_x) * (reference_particle_x - neighbor_particle_x) +
            (reference_particle_y - neighbor_particle_y) * (reference_particle_y - neighbor_particle_y) +
            (reference_particle_z - neighbor_particle_z) * (reference_particle_z - neighbor_particle_z)
        );

        if (norm != 0) {
            float acceleration = compute_acceleration(norm);
            float ax = acceleration * reference_particle_x / norm;
            float ay = acceleration * reference_particle_y / norm;
            float az = acceleration * reference_particle_z / norm;

            reference_particle_ax += ax;
            reference_particle_ay += ay;
            reference_particle_az += az;
        }
    }

    int accelerations_block_idx = ((blockIdx.x * 27 + blockIdx.y) * MAX_PARTICLES_PER_CELL + threadIdx.x) * 3;
    accelerations[accelerations_block_idx] = reference_particle_ax;
    accelerations[accelerations_block_idx + 1] = reference_particle_ay;
    accelerations[accelerations_block_idx + 2] = reference_particle_az;

    return;
}

__global__ particle_update(struct Cell *cell_list, float *accelerations)
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

    int particle_count;
    struct Particle *particle_list;
    struct Cell *cell_list;
    struct Cell *device_cell_list1;
    struct Cell *device_cell_list2;
    float *accelerations;

    import_atoms(argv[1], &particle_list, &particle_count);
    create_cell_list(particle_list, particle_count, cell_list, CELL_CUTOFF_RADIUS_ANGST);
    GPU_PERROR(cudaMalloc(&device_cell_list1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMalloc(&device_cell_list2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell)));
    GPU_PERROR(cudaMalloc(&accelerations, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 27 * MAX_PARTICLES_PER_CELL * 3 * sizeof(float)));
    GPU_PERROR(cudaMemcpy(device_cell_list_1, cell_list, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyHostToDevice));

    dim3 numBlocksCalculate(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z, 27);
    dim3 numBlocksUpdate(CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_CELL);

    int t;
    for (t = 0; t < TIMESTEPS; ++t) {
        GPU_PERROR(cudaMemset(accelerations, 0, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * 27 * MAX_PARTICLES_PER_CELL * 3 * sizeof(float)));
        if (t & 1 == 0) {
            force_eval<<<numBlocksCalculate, threadsPerBlock>>>(device_cell_list1, accelerations);
            particle_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list1, accelerations);
            motion_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list1, device_cell_list2);
        } else {
            force_eval<<<numBlocksCalculate, threadsPerBlock>>>(device_cell_list2, accelerations);
            particle_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list2, accelerations);
            motion_update<<<numBlocksUpdate, threadsPerBlock>>>(device_cell_list2, device_cell_list1);
        }
    }

    if (t & 1) {
        GPU_PERROR(cudaMemcpy(cell_list, device_cell_list2, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(cell_list, device_cell_list1, CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z * sizeof(struct Cell), cudaMemcpyDeviceToHost));
    }

    FILE *out = fopen(argv[2], "w");
    fprintf(out, "cell_idx,particle_id,x,y,z\n");

    GPU_PERROR(cudaFree(device_cell_list1));
    GPU_PERROR(cudaFree(device_cell_list2));

    for (int x = 0; x < CELL_LENGTH_X; ++x) {
        for (int y = 0; y < CELL_LENGTH_Y; ++y) {
            for (int z = 0; z < CELL_LENGTH_Z; ++z) {
                int count = 0;
                struct Cell current_cell = cell_list[x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y];
                while (current_cell.particle_list[count].particle_id != -1) {
                    fprintf(out, "%d,%d,%f,%f,%f\n", x + y * CELL_LENGTH_X + z * CELL_LENGTH_X * CELL_LENGTH_Y,
                                                     current_cell.particle_list[count].particle_id,
                                                     current_cell.particle_list[count].x,
                                                     current_cell.particle_list[count].y,
                                                     current_cell.particle_list[count].z);
                    count++;
                }
            }
        }
    }
}