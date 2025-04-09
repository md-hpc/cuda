#include "pdb_importer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//#define EPSILON (1.65e-9)                       // ng * m^2 / s^2
#define EPSILON (1.65e11)                       // ng * A^2 / s^2 originally (1.65e-9)
#define ARGON_MASS (39.948 * 1.66054e-15)       // ng
#define SIGMA (3.4f)                            // Angstrom
#define R_MIN (3.13796173693f)                  // r(-4 * V(2^(1/6) * EPSILON))
#define LJMAX_ACCELERATION (6.8829688151e25)    // A(R_MIN) in Angstrom / s^2


float compute_acceleration(float r_angstrom) {
        if (r_angstrom < R_MIN)
            return LJMAX_ACCELERATION;

        // in A / s^2
        float temp = SIGMA / r_angstrom;
        temp = temp * temp; // ^2
        temp = temp * temp * temp; // ^ 6

        return 24 * EPSILON * (2 * temp * temp - temp) / (r_angstrom * ARGON_MASS);
        //float force = 4 * EPSILON * (12 * pow(SIGMA, 12.0f) / pow(r, 13.0f) - 6 * pow(SIGMA, 6.0f) / pow(r, 7.0f)) / ARGON_MASS;

}

void naive(int *particle_ids, float *x, float *y, float *z, int particle_count)
{
        float accelerations[particle_count][3];
        float *vx = calloc(particle_count, sizeof(float));
        float *vy = calloc(particle_count, sizeof(float));
        float *vz = calloc(particle_count, sizeof(float));

	FILE *file = fopen("out_naive.csv", "w");
        fprintf(file, "particle_id,x,y,z\n");

        // timestep
        for (unsigned int t = 0; t < TIMESTEPS; ++t) {
                memset(accelerations, 0, sizeof(accelerations));

                // force computation
                for (int i = 0; i < particle_count; ++i) {
                        for (int j = 0; j < particle_count; ++j) {
                                if (i == j)
                                        continue;

                                float diff_x = x[i] - x[j];
                                float diff_y = y[i] - y[j];
                                float diff_z = z[i] - z[j];

                                diff_x += ((diff_x < -UNIVERSE_LENGTH / 2) - (diff_x > UNIVERSE_LENGTH / 2)) * UNIVERSE_LENGTH;
                                diff_y += ((diff_y < -UNIVERSE_LENGTH / 2) - (diff_y > UNIVERSE_LENGTH / 2)) * UNIVERSE_LENGTH;
                                diff_z += ((diff_z < -UNIVERSE_LENGTH / 2) - (diff_z > UNIVERSE_LENGTH / 2)) * UNIVERSE_LENGTH;

                                float norm = sqrt((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));
                                
                                float acceleration = compute_acceleration(norm) / norm;
                                accelerations[i][0] += acceleration * diff_x;
                                accelerations[i][1] += acceleration * diff_y;
                                accelerations[i][2] += acceleration * diff_z;
                
                                //printf("i: %d, a: %f\n", i, acceleration);
                                //printf("i: %d, xyz: (%f, %f, %f)\n", i, x[i], y[i], z[i]);
                        }
                }


                // motion update
                for (int i = 0; i < particle_count; ++i) {
                        vx[i] += accelerations[i][0] * TIMESTEP_DURATION_FS;
                        vy[i] += accelerations[i][1] * TIMESTEP_DURATION_FS;
                        vz[i] += accelerations[i][2] * TIMESTEP_DURATION_FS;

                        // get new reference particle position taking into account periodic boundary conditions
                        float temp_x = x[i] + vx[i] * TIMESTEP_DURATION_FS;
                        temp_x += ((temp_x < 0) - (temp_x > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
                        x[i] = temp_x;
                        
                        float temp_y = y[i] + vy[i] * TIMESTEP_DURATION_FS;
                        temp_y += ((temp_y < 0) - (temp_y > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
                        y[i] = temp_y;

                        float temp_z = z[i] + vz[i] * TIMESTEP_DURATION_FS;
                        temp_z += ((temp_z < 0) - (temp_z > UNIVERSE_LENGTH)) * UNIVERSE_LENGTH;
                        z[i] = temp_z;

                        fprintf(file, "%d,%f,%f,%f\n", particle_ids[i], x[i], y[i], z[i]);
                }
                fprintf(file, "\n");
        }
}
