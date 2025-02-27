#include "pdb_importer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//#define EPSILON (1.65e-9)                       // ng * m^2 / s^2
#define EPSILON (1.65e11)                        // ng * A^2 / s^2
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (0.034f)                           // A
#define LJMIN (-4.0f * 24.0f * EPSILON / SIGMA * (powf(7.0f / 26.0f, 7.0f / 6.0f) - 2.0f * powf(7.0f / 26.0f, 13.0f / 6.0f)))


float compute_acceleration(float r_angstrom) {
        // in A / s^2
        float temp = pow(SIGMA / r_angstrom, 6);
        float acceleration = 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
        //float force = 4 * EPSILON * (12 * pow(SIGMA, 12.0f) / pow(r, 13.0f) - 6 * pow(SIGMA, 6.0f) / pow(r, 7.0f)) / ARGON_MASS;

        //return (force < LJMIN) * LJMIN + !(force < LJMIN) * force;
        return acceleration;
}

void naive(struct Particle *particle_list, const int particle_count)
{
        float accelerations[particle_count][3];

        // timestep
        for (unsigned int t = 0; t < TIMESTEPS; ++t) {
                memset(accelerations, 0, sizeof(accelerations));

                // force computation
                for (int i = 0; i < particle_count; ++i) {
                        for (int j = 0; j < particle_count; ++j) {
                                if (i == j)
                                        continue;

                                struct Particle reference_particle = particle_list[i];
                                struct Particle neighbor_particle = particle_list[j];

                                float norm = sqrt(
                                        pow(reference_particle.x - neighbor_particle.x, 2) +
                                        pow(reference_particle.y - neighbor_particle.y, 2) +
                                        pow(reference_particle.z - neighbor_particle.z, 2)
                                );
                                
                                float acceleration = compute_acceleration(norm);
                                accelerations[i][0] += acceleration * (reference_particle.x - neighbor_particle.x) / norm;
                                accelerations[i][1] += acceleration * (reference_particle.y - neighbor_particle.y) / norm;
                                accelerations[i][2] += acceleration * (reference_particle.z - neighbor_particle.z) / norm;
                        }
                }

/*
                for (int i = 0; i < 10; ++i) {
                        printf("accelerations: %.30f\t%.30f\t%.30f\n", accelerations[i][0] * TIMESTEP_DURATION_FS, accelerations[i][1] * TIMESTEP_DURATION_FS, accelerations[i][2] * TIMESTEP_DURATION_FS);
                }
*/

                // motion update
                for (int i = 0; i < particle_count; ++i) {
                        struct Particle reference_particle = particle_list[i];

                        reference_particle.vx += accelerations[i][0] * TIMESTEP_DURATION_FS;
                        reference_particle.vy += accelerations[i][1] * TIMESTEP_DURATION_FS;
                        reference_particle.vz += accelerations[i][2] * TIMESTEP_DURATION_FS;

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

                        particle_list[i] = reference_particle;
                }
        }
}
