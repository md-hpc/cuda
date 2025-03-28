extern "C" {

#include "pdb_importer.h"

}
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <assert.h>

#define MAX_PARTICLES_PER_BLOCK 32
#define CELL_CUTOFF_RADIUS_ANGST 100
//#define EPSILON (1.65e-9)                       // ng * m^2 / s^2
#define EPSILON (1.65e11)                        // ng * A^2 / s^2
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (0.034f)                           // A
#define LJMIN (-4.0f * 24.0f * EPSILON / SIGMA * (powf(7.0f / 26.0f, 7.0f / 6.0f) - 2.0f * powf(7.0f / 26.0f, 13.0f / 6.0f)))
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);


__device__ float compute_acceleration(float r_angstrom) {
        // in A / s^2
        float temp = powf(SIGMA / r_angstrom, 6);
        float acceleration = 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
        //float force = 4 * EPSILON * (12 * pow(SIGMA, 12.0f) / pow(r, 13.0f) - 6 * pow(SIGMA, 6.0f) / pow(r, 7.0f)) / ARGON_MASS;

        //return (force < LJMIN) * LJMIN + !(force < LJMIN) * force;
        return acceleration;
}

__global__ void calculate_accelerations(struct Particle *src_particle_list, int particle_count, float *accelerations)
{
    __shared__ struct Particle shared_particles[MAX_PARTICLES_PER_BLOCK];

    // each thread gets a particle as a reference particle
    const int reference_particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reference_particle_idx >= particle_count)
        return; 

    const int accelerations_block_idx = blockIdx.x * particle_count * sizeof(float) * 3;
    struct Particle reference_particle = src_particle_list[reference_particle_idx];
    float ax = 0;
    float ay = 0;
    float az = 0;

    // loop through particles MAX_PARTICLES_PER_BLOCK at at time
    int i;
    for (i = 0; i < particle_count - MAX_PARTICLES_PER_BLOCK + 1; i += MAX_PARTICLES_PER_BLOCK) {
        shared_particles[threadIdx.x] = src_particle_list[i + threadIdx.x];

        //__syncthreads();

        for (int j = 0; j < MAX_PARTICLES_PER_BLOCK; ++j) {
            struct Particle neighbor_particle = shared_particles[(threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK];

            if (reference_particle.particle_id == neighbor_particle.particle_id)
                continue;

            float norm = sqrt(
                pow(reference_particle.x - neighbor_particle.x, 2) +
                pow(reference_particle.y - neighbor_particle.y, 2) +
                pow(reference_particle.z - neighbor_particle.z, 2)
            );
            
            float acceleration = compute_acceleration(norm);
            ax += acceleration * (reference_particle.x - neighbor_particle.x) / norm;
            ay += acceleration * (reference_particle.y - neighbor_particle.y) / norm;
            az += acceleration * (reference_particle.z - neighbor_particle.z) / norm;

            if (reference_particle.particle_id < neighbor_particle.particle_id) {
                accelerations[accelerations_block_idx + ((threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK) * 3] -= acceleration * (reference_particle.x - neighbor_particle.x) / norm;
                accelerations[accelerations_block_idx + ((threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK) * 3 + 1] -= acceleration * (reference_particle.y - neighbor_particle.y) / norm;
                accelerations[accelerations_block_idx + ((threadIdx.x + j) % MAX_PARTICLES_PER_BLOCK) * 3 + 2] -= acceleration * (reference_particle.z - neighbor_particle.z) / norm;
            }

            //__syncthreads();
        }
    }

    int remaining_particles = particle_count - i;
    if (threadIdx.x < remaining_particles)
        shared_particles[threadIdx.x] = src_particle_list[i + threadIdx.x];
    //__syncthreads();

    for (; i < remaining_particles; ++i) {
        struct Particle neighbor_particle = shared_particles[i];

        if (reference_particle.particle_id == neighbor_particle.particle_id)
            continue;

        float norm = sqrt(
            pow(reference_particle.x - neighbor_particle.x, 2) +
            pow(reference_particle.y - neighbor_particle.y, 2) +
            pow(reference_particle.z - neighbor_particle.z, 2)
        );
        
        float acceleration = compute_acceleration(norm);
        ax += acceleration * (reference_particle.x - neighbor_particle.x) / norm;
        ay += acceleration * (reference_particle.y - neighbor_particle.y) / norm;
        az += acceleration * (reference_particle.z - neighbor_particle.z) / norm;

        if (reference_particle.x < neighbor_particle.x) {
            atomicAdd(&accelerations[accelerations_block_idx + i * 3], -acceleration * (reference_particle.x - neighbor_particle.x) / norm);
            atomicAdd(&accelerations[accelerations_block_idx + i * 3 + 1], -acceleration * (reference_particle.y - neighbor_particle.y) / norm);
            atomicAdd(&accelerations[accelerations_block_idx + i * 3 + 2], -acceleration * (reference_particle.z - neighbor_particle.z) / norm);
        }
    }

    // each thread writes reference particle accelerations back to global memory
    accelerations[accelerations_block_idx + threadIdx.x * 3] = ax;
    accelerations[accelerations_block_idx + threadIdx.x * 3 + 1] = ay;
    accelerations[accelerations_block_idx + threadIdx.x * 3 + 2] = az;
}

__global__ void position_update(struct Particle *src_particle_list, struct Particle *dst_particle_list, int particle_count, float *accelerations)
{
    int reference_particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reference_particle_idx >= particle_count)
        return;

    struct Particle reference_particle = src_particle_list[reference_particle_idx];

    float ax = 0;
    float ay = 0;
    float az = 0;

    const int accelerations_block_size = particle_count * sizeof(float) * 3;
    for (int i = 0; i < gridDim.x; ++i) {
        ax += accelerations[reference_particle_idx + accelerations_block_size * i];
        ay += accelerations[reference_particle_idx + accelerations_block_size * i + 1];
        az += accelerations[reference_particle_idx + accelerations_block_size * i + 2];
    }

    // calculate velocity for reference particle
    reference_particle.vx += ax * TIMESTEP_DURATION_FS;
    reference_particle.vy += ay * TIMESTEP_DURATION_FS;
    reference_particle.vz += az * TIMESTEP_DURATION_FS;

    // get new reference particle position taking into account periodic boundary conditions
    float x = reference_particle.x + reference_particle.vx * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
    reference_particle.x = x;

    float y = reference_particle.y + reference_particle.vy * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
    reference_particle.y = y;

    float z = reference_particle.z + reference_particle.vz * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
    reference_particle.z = z;

    dst_particle_list[reference_particle_idx] = reference_particle;
}

int main(int argc, char **argv) 
{
    if (argc != 3) {
        printf("Usage: ./nsquared <input_file> <output_file>\n");
        return 1; 
    }
    
    char *input_file = argv[1];
    char *output_file = argv[2];

    int particle_count;
    struct Particle *particle_list;
    struct Particle *device_particle_list_1;
    struct Particle *device_particle_list_2;
    float *accelerations;
    import_atoms(input_file, &particle_list, &particle_count);

    GPU_PERROR(cudaMalloc(&accelerations, ((particle_count - 1) / MAX_PARTICLES_PER_BLOCK + 1) * particle_count * sizeof(float) * 3));
    GPU_PERROR(cudaMalloc(&device_particle_list_1, particle_count * sizeof(struct Particle)));
    GPU_PERROR(cudaMalloc(&device_particle_list_2, particle_count * sizeof(struct Particle)));
    GPU_PERROR(cudaMemcpy(device_particle_list_1, particle_list, particle_count * sizeof(struct Particle), cudaMemcpyHostToDevice));

    // set parameters
    dim3 numBlocks((particle_count - 1) / MAX_PARTICLES_PER_BLOCK + 1);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BLOCK);
    struct Particle *buff = (struct Particle *) malloc(particle_count * sizeof(struct Particle));

#ifdef SIMULATE
    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
#endif

#ifdef TIME_RUN
    struct timespec time_start;
    struct timespec time_stop;
    clock_gettime(CLOCK_REALTIME, &time_start);
#endif

    for (int t = 1l; t <= TIMESTEPS; ++t) {
        GPU_PERROR(cudaMemset(accelerations, 0, ((particle_count - 1) / MAX_PARTICLES_PER_BLOCK + 1) * particle_count * sizeof(float) * 3));
        if (t % 2 == 1) {
            calculate_accelerations<<<numBlocks, threadsPerBlock>>>(device_particle_list_1, particle_count, accelerations);
            position_update<<<numBlocks, threadsPerBlock>>>(device_particle_list_1, device_particle_list_2, particle_count, accelerations);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(buff, device_particle_list_2, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
#endif
        } else {
            calculate_accelerations<<<numBlocks, threadsPerBlock>>>(device_particle_list_2, particle_count, accelerations);
            position_update<<<numBlocks, threadsPerBlock>>>(device_particle_list_2, device_particle_list_1, particle_count, accelerations);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(buff, device_particle_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
#endif
        }
#ifdef SIMULATE
        for (int i = 0; i < particle_count; ++i) {
            fprintf(out, "%d,%f,%f,%f\n", buff[i].particle_id, buff[i].x, buff[i].y, buff[i].z);
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

    printf("nsquared_n3l,%f\n", ((double) temp.tv_sec) + (((double) temp.tv_nsec) * 1e-9));

    struct Particle *out_list = (struct Particle *) malloc(particle_count * sizeof(struct Particle));
    if (TIMESTEPS & 1) {
        GPU_PERROR(cudaMemcpy(out_list, device_particle_list_2, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(out_list, device_particle_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
    }
        
    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
    for (int i = 0; i < particle_count; ++i) {
        fprintf(out, "%d,%f,%f,%f\n", out_list[i].particle_id, out_list[i].x, out_list[i].y, out_list[i].z);
    }
    free(out_list);
#endif

    GPU_PERROR(cudaFree(device_particle_list_1));
    GPU_PERROR(cudaFree(device_particle_list_2));

    return 0;
}
