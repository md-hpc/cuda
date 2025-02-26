#include "pdb_importer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPSILON (1.65e-21)
#define ARGON_MASS (39.948 * 1.66054e-27)
#define SIGMA 0.34f
#define LJMIN (-4.0f * 24.0f * EPSILON / SIGMA * (powf(7.0f / 26.0f, 7.0f / 6.0f) - 2.0f * powf(7.0f / 26.0f, 13.0f / 6.0f)))


float compute_acceleration(float r) {
    float force = 4 * EPSILON * (12 * powf(SIGMA, 12.0f) / powf(r, 13.0f) - 6 * powf(SIGMA, 6.0f) / powf(r, 7.0f)) / ARGON_MASS;

    return (force < LJMIN) * LJMIN + !(force < LJMIN) * force;
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