extern "C" {

#include "pdb_importer.h"

}
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <assert.h>

#define MAX_PARTICLES_PER_BLOCK 1024
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

__global__ void timestep(struct Particle *src_particle_list, struct Particle *dst_particle_list, int particle_count)
{
    // each thread gets a particle as a reference particle
    int reference_particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reference_particle_idx >= particle_count)
        return; 

    struct Particle reference_particle = src_particle_list[reference_particle_idx];

    float ax = 0;
    float ay = 0;
    float az = 0;

    // accumulate accelerations for every other particle
    for (int i = 1; i < particle_count; ++i) {
        struct Particle neighbor_particle = src_particle_list[(reference_particle_idx + i) % particle_count];

        float norm = sqrtf(
            powf(reference_particle.x - neighbor_particle.x, 2) +
            powf(reference_particle.y - neighbor_particle.y, 2) +
            powf(reference_particle.z - neighbor_particle.z, 2)
        );
        
        float acceleration = compute_acceleration(norm);
        ax += acceleration * (reference_particle.x - neighbor_particle.x) / norm;
        ay += acceleration * (reference_particle.y - neighbor_particle.y) / norm;
        az += acceleration * (reference_particle.z - neighbor_particle.z) / norm;
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
    import_atoms(input_file, &particle_list, &particle_count);

    GPU_PERROR(cudaMalloc(&device_particle_list_1, particle_count * sizeof(struct Particle)));
    GPU_PERROR(cudaMalloc(&device_particle_list_2, particle_count * sizeof(struct Particle)));
    GPU_PERROR(cudaMemcpy(device_particle_list_1, particle_list, particle_count * sizeof(struct Particle), cudaMemcpyHostToDevice));

    // set parameters
    dim3 numBlocks((particle_count - 1) / MAX_PARTICLES_PER_BLOCK + 1);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BLOCK);
    struct Particle *buff = (struct Particle *) malloc(particle_count * sizeof(struct Particle));
    GPU_PERROR(cudaMemcpy(buff, device_particle_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));

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
        if (t % 2 == 1) {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_list_1, device_particle_list_2, particle_count);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(buff, device_particle_list_2, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
#endif
        } else {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_list_2, device_particle_list_1, particle_count);
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

    printf("nsquared,%f\n", ((double) temp.tv_sec) + (((double) temp.tv_nsec) * 1e-9));

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
