extern "C" {

#include "pdb_importer.h"

}
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <assert.h>

#define MAX_PARTICLES_PER_BLOCK 1024
#define CELL_CUTOFF_RADIUS_ANGST 100
#define EPSILON (1.65e-21)
#define ARGON_MASS (39.948 * 1.66054e-27)
#define SIGMA 0.34f
#define LJMIN (-4.0f * 24.0f * EPSILON / SIGMA * (powf(7.0f / 26.0f, 7.0f / 6.0f) - 2.0f * powf(7.0f / 26.0f, 13.0f / 6.0f)))
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);


__device__ float compute_acceleration(float r) {
    float force = 4 * EPSILON * (12 * powf(SIGMA, 12.0f) / powf(r, 13.0f) - 6 * powf(SIGMA, 6.0f) / powf(r, 7.0f)) / ARGON_MASS;

    return (force < LJMIN) * LJMIN + !(force < LJMIN) * force;
}

__global__ void timestep(struct Particle *src_particle_list, struct Particle *dst_particle_list, int particle_count)
{
    // each thread gets a particle as a reference particle
    int reference_particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reference_particle_idx >= particle_count) return; 

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
	ax += acceleration * reference_particle.x / norm;
	ay += acceleration * reference_particle.y / norm;
	az += acceleration * reference_particle.z / norm;
    }

    // calculate velocity for reference particle
    reference_particle.vx += ax * TIMESTEP_DURATION_FS;
    reference_particle.vy += ay * TIMESTEP_DURATION_FS;
    reference_particle.vz += az * TIMESTEP_DURATION_FS;

    // get new reference particle position taking into account periodic boundary conditions
    float x = reference_particle.x + reference_particle.vx * TIMESTEP_DURATION_FS;
    x += ((x < 0) - (x > CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_X * CELL_CUTOFF_RADIUS_ANGST);
    reference_particle.x = x;
 
    float y = reference_particle.y + reference_particle.vy * TIMESTEP_DURATION_FS;
    y += ((y < 0) - (y > CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Y * CELL_CUTOFF_RADIUS_ANGST);
    reference_particle.y = y;

    float z = reference_particle.z + reference_particle.vz * TIMESTEP_DURATION_FS;
    z += ((z < 0) - (z > CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST)) * (CELL_LENGTH_Z * CELL_CUTOFF_RADIUS_ANGST);
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
    dim3 numBlocks((particle_count - 1) / 1024 + 1);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BLOCK);
    struct Particle *buff = (struct Particle *) malloc(particle_count * sizeof(struct Particle));
    GPU_PERROR(cudaMemcpy(buff, device_particle_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));

    for (int t = 1l; t <= TIMESTEPS; ++t) {
        
        if (t % 2 == 1) {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_list_1, device_particle_list_2, particle_count);
            GPU_PERROR(cudaMemcpy(buff, device_particle_list_2, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
        } else {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_list_2, device_particle_list_1, particle_count);
            GPU_PERROR(cudaMemcpy(buff, device_particle_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
        }
    }

    struct Particle *out_list = (struct Particle *) malloc(particle_count * sizeof(struct Particle));
    if (TIMESTEPS & 1) {
        GPU_PERROR(cudaMemcpy(out_list, device_particle_list_2, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
    } else {
        GPU_PERROR(cudaMemcpy(out_list, device_particle_list_1, particle_count * sizeof(struct Particle), cudaMemcpyDeviceToHost));
    }
        
    GPU_PERROR(cudaFree(device_particle_list_1));
    GPU_PERROR(cudaFree(device_particle_list_2));

    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
    for (int i = 0; i < particle_count; ++i) {
        fprintf(out, "%d,%f,%f,%f\n", out_list[i].particle_id, out_list[i].x, out_list[i].y, out_list[i].z);
    }
    free(out_list);

    return 0;
}
