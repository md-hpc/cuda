#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_PARTICLES 5
#define MAX_DIMENSION_LENGTH 10         // universe length in one dimension
#define TIME_STEPS 1
#define DT 1                            // amount of time per time step
#define EPSILON 1
#define SIGMA 1

struct Particle {
        int particleID;
        float x;
        float y;
        float z;
        float vx;
        float vy;
        float vz;
};

void init_particles(struct Particle particleList[NUM_PARTICLES])
{
        for (int i = 0; i < NUM_PARTICLES; ++i) {
                particleList[i].particleID = i;
                particleList[i].x = rand() % MAX_DIMENSION_LENGTH;
                particleList[i].y = rand() % MAX_DIMENSION_LENGTH;
                particleList[i].z = rand() % MAX_DIMENSION_LENGTH;
                particleList[i].vx = 0;
                particleList[i].vy = 0;
                particleList[i].vz = 0;
        }
}

float force_computation(float distance1, float distance2)
{
        float norm = sqrt(distance1 * distance1 + distance2 * distance2);
        float temp = pow(SIGMA / norm, 6);

        return EPSILON * (48 * temp * temp - 24 * temp) / norm;
}

int main()
{
        struct Particle particleList[NUM_PARTICLES];
        init_particles(particleList);
        for (int i = 0; i < NUM_PARTICLES; ++i) {
                printf("particle %d: (%f, %f, %f)\n", i, particleList[i].x, particleList[i].y, particleList[i].z);
        }
        printf("\n");

        float accelerations[NUM_PARTICLES][3];

        // timestep
        for (int t = 0; t < TIME_STEPS; ++t) {
                memset(accelerations, 0, sizeof(accelerations));

                // force computation
                for (int i = 0; i < NUM_PARTICLES; ++i) {
                        for (int j = 0; j < NUM_PARTICLES; ++j) {
                                accelerations[i][0] += force_computation(particleList[i].x, particleList[j].x) * DT;
                                accelerations[i][1] += force_computation(particleList[i].y, particleList[j].y) * DT;
                                accelerations[i][2] += force_computation(particleList[i].z, particleList[j].z) * DT;
                        }
                }

                // motion update
                for (int i = 0; i < NUM_PARTICLES; ++i) {
                        particleList[i].vx += accelerations[i][0] * DT;
                        particleList[i].vy += accelerations[i][1] * DT;
                        particleList[i].vz += accelerations[i][2] * DT;

                        particleList[i].x = (particleList[i].x + particleList[i].vx * DT) - MAX_DIMENSION_LENGTH * floor(particleList[i].x / MAX_DIMENSION_LENGTH);
                        particleList[i].y = (particleList[i].y + particleList[i].vy * DT) - MAX_DIMENSION_LENGTH * floor(particleList[i].y / MAX_DIMENSION_LENGTH);
                        particleList[i].z = (particleList[i].z + particleList[i].vz * DT) - MAX_DIMENSION_LENGTH * floor(particleList[i].z / MAX_DIMENSION_LENGTH);
                }
        }

        for (int i = 0; i < NUM_PARTICLES; ++i) {
                printf("particle %d: (%f, %f, %f)\n", i, particleList[i].x, particleList[i].y, particleList[i].z);
        }

        return 0;
}