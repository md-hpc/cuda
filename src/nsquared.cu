extern "C" {

#include "pdb_importer.h"

}
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <assert.h>
#include <math.h>

#define MAX_PARTICLES_PER_BLOCK 1024
#define EPSILON (1.65e11)                       // ng * A^2 / s^2 originally (1.65e-9)
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (0.034f)                          // A
#define GPU_PERROR(err) do {\
    if (err != cudaSuccess) {\
        fprintf(stderr,"gpu_perror: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(err);\
    }\
} while (0);

#define R_MIN (0.0317782790163)
#define LJMAX_ACCELERATION (24.7987876375)
//constexpr float LJMAX = (4.0f * 24.0f * EPSILON / SIGMA * (0.216344308307f - 2.0f * 0.0582465445441f));
#ifdef VALIDATE
__device__ float compute_potential(float r_angstrom) {
        // validation code for getting potential energy
        float temp = SIGMA / r_angstrom;
        temp = temp * temp; // ^2
        temp = temp * temp * temp; // ^ 6
        float potential = 4 * EPSILON * ((temp * temp) - temp);
        return potential;
}
#endif

__device__ float compute_acceleration(float r_angstrom) {
        if (r_angstrom < R_MIN)
            return LJMAX_ACCELERATION;

        // in A / s^2
        float temp = SIGMA / r_angstrom;
        temp = temp * temp; // ^2
        temp = temp * temp * temp; // ^ 6

        return 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
        //float force = 4 * EPSILON * (12 * pow(SIGMA, 12.0f) / pow(r, 13.0f) - 6 * pow(SIGMA, 6.0f) / pow(r, 7.0f)) / ARGON_MASS;

}

__global__ void timestep(float *particle_id, float *src_x, float *src_y, float *src_z,
                         float *vx, float *vy, float *vz, float *dst_x, float *dst_y,
                         float *dst_z, int particle_count, float *device_pe)
{
    // each thread gets a particle as a reference particle
    int reference_particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // extra threads can exit 
    if (reference_particle_idx >= particle_count)
        return; 

    // get reference particle positions
    float reference_x = src_x[reference_particle_idx]; 
    float reference_y = src_y[reference_particle_idx]; 
    float reference_z = src_z[reference_particle_idx]; 

    // accumulate accelerations for every other particle (i == 1)
    float ax = 0;
    float ay = 0;
    float az = 0;

    for (int i = 1; i < particle_count; ++i) {
        // use temp variables to optimize
        float diff_x = reference_x - src_x[(reference_particle_idx + i) % particle_count];
        float diff_y = reference_y - src_y[(reference_particle_idx + i) % particle_count];
        float diff_z = reference_z - src_z[(reference_particle_idx + i) % particle_count];

        // get new particle position differences taking into account periodic boundary conditions
        diff_x += ((diff_x < -UNIVERSE_LENGTH / 2) - (diff_x > UNIVERSE_LENGTH / 2)) * UNIVERSE_LENGTH;
        diff_y += ((diff_y < -UNIVERSE_LENGTH / 2) - (diff_y > UNIVERSE_LENGTH / 2)) * UNIVERSE_LENGTH;
        diff_z += ((diff_z < -UNIVERSE_LENGTH / 2) - (diff_z > UNIVERSE_LENGTH / 2)) * UNIVERSE_LENGTH;

        // get norm for acceleration calculation
        float norm = sqrtf((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));

        // compute scalar acceleration and apply to xyz directions 
        float acceleration = compute_acceleration(norm) / norm;
        ax += acceleration * diff_x;
        ay += acceleration * diff_y;
        az += acceleration * diff_z;
#ifdef VALIDATE
        device_pe[i] = compute_potential(norm);
#endif
    }

    // obtain current velocity of reference particle
    float reference_vx = vx[reference_particle_idx]; 
    float reference_vy = vy[reference_particle_idx]; 
    float reference_vz = vz[reference_particle_idx]; 
    // calculate velocity for reference particle
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
        printf("Usage: ./nsquared <input_file> <output_file>\n");
        return 1; 
    }
    
    char *input_file = argv[1];
    char *output_file = argv[2];

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
    float *device_pe;

    import_atoms(input_file, &host_particle_ids, &host_x, &host_y, &host_z, &particle_count);
    float *host_vx = (float *)malloc(particle_count * sizeof(float));
    float *host_vy = (float *)malloc(particle_count * sizeof(float));
    float *host_vz = (float *)malloc(particle_count * sizeof(float));
    float *host_pe = (float *)malloc(particle_count * sizeof(float));

    GPU_PERROR(cudaMalloc(&device_particle_ids, particle_count * sizeof(int)));
    GPU_PERROR(cudaMalloc(&device_x_1, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_y_1, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_z_1, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_x_2, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_y_2, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_z_2, particle_count * sizeof(float)));
    // need for validation
    GPU_PERROR(cudaMalloc(&vx, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&vy, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&vz, particle_count * sizeof(float)));
    GPU_PERROR(cudaMalloc(&device_pe, particle_count * sizeof(float)));

    GPU_PERROR(cudaMemcpy(device_particle_ids, host_particle_ids, particle_count * sizeof(int), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemcpy(device_x_1, host_x, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemcpy(device_y_1, host_y, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemcpy(device_z_1, host_z, particle_count * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PERROR(cudaMemset(vx, 0.0f, particle_count * sizeof(float)));
    GPU_PERROR(cudaMemset(vy, 0.0f, particle_count * sizeof(float)));
    GPU_PERROR(cudaMemset(vz, 0.0f, particle_count * sizeof(float)));

    // set parameters
    dim3 numBlocks((particle_count - 1) / MAX_PARTICLES_PER_BLOCK + 1);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BLOCK);

#ifdef SIMULATE
    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
#endif

#ifdef TIME_RUN
    struct timespec time_start;
    struct timespec time_stop;
    clock_gettime(CLOCK_REALTIME, &time_start);
#endif

#ifdef VALIDATE
    float momentum_x = 0;
    float momentum_y = 0;
    float momentum_z = 0; 

    float potential_energy = 0;
    float kinetic_energy = 0;
#endif

    for (int t = 1l; t <= TIMESTEPS; ++t) {
        if (t % 2 == 1) {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_ids, device_x_1, device_y_1, device_z_1, vx, vy, vz, device_x_2, device_y_2, device_z_2, particle_count, device_pe);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_x, device_x_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_y, device_y_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_z, device_z_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
#endif
#ifdef VALIDATE
            GPU_PERROR(cudaMemcpy(host_pe, device_pe, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_vx, vx, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_vy, vy, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_vz, vz, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
#endif
        } else {
            timestep<<<numBlocks, threadsPerBlock>>>(device_particle_ids, device_x_2, device_y_2, device_z_2, vx, vy, vz, device_x_1, device_y_1, device_z_1, particle_count, device_pe);
#ifdef SIMULATE
            GPU_PERROR(cudaMemcpy(host_x, device_x_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_y, device_y_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_z, device_z_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
#endif
#ifdef VALIDATE
            GPU_PERROR(cudaMemcpy(host_pe, device_pe, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_vx, vx, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_vy, vy, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
            GPU_PERROR(cudaMemcpy(host_vz, vz, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
#endif
        }
#ifdef SIMULATE
        for (int i = 0; i < particle_count; ++i) {
            fprintf(out, "%d,%f,%f,%f\n", host_particle_ids[i], host_x[i], host_y[i], host_z[i]);
        }
        fprintf(out, "\n");
#endif
#ifdef VALIDATE
        for (int i = 0; i < particle_count; ++i) {
            // accumulate momentum
            momentum_x += host_vx[i];
            momentum_y += host_vy[i];
            momentum_z += host_vz[i];
            // calculate kinetic energy through velocity
            float particle_kinetic_energy = 0.5 * ARGON_MASS * ((host_vx[i] * host_vx[i]) + (host_vy[i] * host_vy[i]) + (host_vz[i] * host_vz[i]));
            potential_energy += host_pe[i];
            kinetic_energy += particle_kinetic_energy;
        }
        printf("potential %.12f + kinetic %.12f = total energy %.12f\n", potential_energy, kinetic_energy, potential_energy + kinetic_energy);
        printf("momentum x: %.12f\n", momentum_x);
        printf("momentum y: %.12f\n", momentum_y);
        printf("momentum z: %.12f\n", momentum_z);
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

    if (TIMESTEPS & 1) {
        GPU_PERROR(cudaMemcpy(host_x, device_x_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_y, device_y_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_z, device_z_2, particle_count * sizeof(float), cudaMemcpyDeviceToHost));


    } else {
        GPU_PERROR(cudaMemcpy(host_x, device_x_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_y, device_y_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        GPU_PERROR(cudaMemcpy(host_z, device_z_1, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    FILE *out = fopen(output_file, "w");
    fprintf(out, "particle_id,x,y,z\n");
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
