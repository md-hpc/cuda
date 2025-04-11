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

#define MAX_PARTICLES_PER_BLOCK 1024
#define EPSILON (1.65e11)                       // ng * A^2 / s^2 originally (1.65e-9)
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (3.4f)                            // Angstrom
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

__global__ void timestep(float *particle_id, float *src_x, float *src_y, float *src_z,
                         float *vx, float *vy, float *vz, float *dst_x, float *dst_y,
                         float *dst_z, int particle_count)
{
    // initialize shared memory - shared between threads in each block
    __shared__ float shared_id[MAX_PARTICLES_PER_BLOCK];
    __shared__ float shared_x[MAX_PARTICLES_PER_BLOCK];
    __shared__ float shared_y[MAX_PARTICLES_PER_BLOCK];
    __shared__ float shared_z[MAX_PARTICLES_PER_BLOCK];

    // each thread gets a particle as a reference particle
    int reference_particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // extra threads can exit
    if (reference_particle_idx >= particle_count)
        return; 

    // get reference particle positions and id
    float reference_particle_id = particle_id[reference_particle_idx];
    float reference_x = src_x[reference_particle_idx]; 
    float reference_y = src_y[reference_particle_idx]; 
    float reference_z = src_z[reference_particle_idx]; 

    // accumulate accelerations for every other particle
    float ax = 0;
    float ay = 0;
    float az = 0;
    for (int i = 0; i < particle_count; i += MAX_PARTICLES_PER_BLOCK) {
        shared_id[threadIdx.x] = particle_id[i + threadIdx.x];
        shared_x[threadIdx.x] = src_x[i + threadIdx.x];
        shared_y[threadIdx.x] = src_y[i + threadIdx.x];
        shared_z[threadIdx.x] = src_z[i + threadIdx.x];

#if MAX_PARTICLES_PER_BLOCK > 32
        __syncthreads();
#endif

        for (int j = 0; j < MAX_PARTICLES_PER_BLOCK; ++j) {
            // dont calculate acceleration with itself
            if (shared_id[(threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK] == reference_particle_id)
                continue;
            // get neighbor particle positions from shared memory
            float neighbor_x = shared_x[(threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK];
            float neighbor_y = shared_y[(threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK];
            float neighbor_z = shared_z[(threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK];
            // use temp variables to optimize
            float diff_x = reference_x - neighbor_x;
            float diff_y = reference_y - neighbor_y;
            float diff_z = reference_z - neighbor_z;
            // get norm for acceleration calculation
            float norm = sqrtf((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));

            // compute scalar acceleration and apply to xyz directions 
            float acceleration = compute_acceleration(norm);
            ax += acceleration * (reference_x - neighbor_x) / norm;
            ay += acceleration * (reference_y - neighbor_y) / norm;
            az += acceleration * (reference_z - neighbor_z) / norm;
        }
#if MAX_PARTICLES_PER_BLOCK > 32
        __syncthreads();
#endif
    }

    // calculate velocity for reference particle
    float reference_vx = vx[reference_particle_idx]; 
    float reference_vy = vy[reference_particle_idx]; 
    float reference_vz = vz[reference_particle_idx]; 
    reference_vx += ax * TIMESTEP_DURATION_FS;
    reference_vy += ay * TIMESTEP_DURATION_FS;
    reference_vz += az * TIMESTEP_DURATION_FS;

    // get new reference particle position taking into account periodic boundary conditions
    float x = reference_x + reference_vx * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
    reference_x = x;
 
    float y = reference_y + reference_vy * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
    reference_y = y;

    float z = reference_z + reference_vz * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
    reference_z = z;

    // write velocity and positions of particle back to global memory
    vx[reference_particle_idx] = reference_vx;
    vy[reference_particle_idx] = reference_vy;
    vz[reference_particle_idx] = reference_vz;
    dst_x[reference_particle_idx] = reference_x;
    dst_y[reference_particle_idx] = reference_y;
    dst_z[reference_particle_idx] = reference_z;
}

int main(int argc, char **argv) 
{
    if (argc != 3) {
        printf("Usage: ./nsquared_shared <input_file> <output_file>\n");
        return 1; 
    }
    
    char *input_file = argv[1];
    char *output_file = argv[2];
    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");

    int particle_count;

    int *host_particle_ids = NULL;
    float *host_x = NULL;
    float *host_y = NULL;
    float *host_z = NULL;

    float *device_particle_ids;
    float *device_x_1;
    float *device_y_1;
    float *device_z_1;
    float *device_x_2;
    float *device_y_2;
    float *device_z_2;
    float *vx;
    float *vy;
    float *vz;

    import_atoms(input_file, &host_particle_ids, &host_x, &host_y, &host_z, &particle_count);

    GPU_PERROR(cudaMalloc(&device_particle_ids, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_x_1, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_y_1, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_z_1, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_x_2, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_y_2, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_z_2, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&vx, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&vy, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&vz, particle_count * sizeof(float)));

    GPU_PERROR(cudaMemcpy(device_particle_ids, host_particle_ids, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemcpy(device_x_1, host_x, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemcpy(device_y_1, host_y, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemcpy(device_z_1, host_z, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemset(vx, 0, particle_count * sizeof(float)));
    GPU_PERROR(cudaMemset(vy, 0, particle_count * sizeof(float)));
    GPU_PERROR(cudaMemset(vz, 0, particle_count * sizeof(float)));

    // set parameters
    dim3 numBlocks((particle_count - 1) / MAX_PARTICLES_PER_BLOCK + 1);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BLOCK);

#ifdef TIME_RUN
    cudaEvent_t time_start;
    cudaEvent_t time_stop;
    GPU_PERROR(cudaEventCreate(&time_start));
    GPU_PERROR(cudaEventCreate(&time_stop));

    GPU_PERROR(cudaEventRecord(time_start));
#endif

    for (int t = 0; t < TIMESTEPS; ++t) {
        if (t % 2 == 0) {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_ids, device_x_1, device_y_1, device_z_1, vx, vy, vz, device_x_2, device_y_2, device_z_2, particle_count);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_x, device_x_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_y, device_y_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_z, device_z_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
#endif
        } else {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_ids, device_x_2, device_y_2, device_z_2, vx, vy, vz, device_x_1, device_y_1, device_z_1, particle_count);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_x, device_x_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_y, device_y_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_z, device_z_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
#endif
        }
#ifdef SIMULATE
        for (int i = 0; i < particle_count; ++i) {
            fprintf(out, "%d,%f,%f,%f\n", host_particle_ids[i], host_x[i], host_y[i], host_z[i]);
        }
        fprintf(out, "\n");
#endif
    }

#ifdef TIME_RUN
    GPU_PERROR(cudaEventRecord(time_stop));

    if (TIMESTEPS % 2 == 1) {
        GPU_PERROR(cudaMemcpy(host_x, device_x_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_y, device_y_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_z, device_z_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(host_x, device_x_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_y, device_y_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_z, device_z_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    GPU_PERROR(cudaEventSynchronize(time_stop));
    float elapsed_milliseconds = 0;
    GPU_PERROR(cudaEventElapsedTime(&elapsed_milliseconds, time_start, time_stop));
    printf("nsquared_shared,%d,%f\n", particle_count, elapsed_milliseconds / 1000);
        
    for (int i = 0; i < particle_count; ++i) {
        fprintf(out, "%d,%f,%f,%f\n", host_particle_ids[i], host_x[i], host_y[i], host_z[i]);
    }
#endif

    GPU_PERROR(cudaFree(device_particle_ids));
    GPU_PERROR(cudaFree(device_x_1));
    GPU_PERROR(cudaFree(device_y_1));
    GPU_PERROR(cudaFree(device_z_1));
    GPU_PERROR(cudaFree(device_x_2));
    GPU_PERROR(cudaFree(device_y_2));
    GPU_PERROR(cudaFree(device_z_2));
    GPU_PERROR(cudaFree(vx));
    GPU_PERROR(cudaFree(vy));
    GPU_PERROR(cudaFree(vz));

    return 0;
}
