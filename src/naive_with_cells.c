#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_PARTICLES 5
#define MAX_PARTICLES_PER_CELL 120

#define CELL_CUTOFF_RADIUS 1f
#define CELL_LENGTH_X 3
#define CELL_LENGTH_Y 3
#define CELL_LENGTH_Z 3

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

struct Cell {
        struct Particle particleList[MAX_PARTICLES_PER_CELL];
        int next;
};

void init(struct Cell cellList[CELL_LENGTH_X][CELL_LENGTH_Y][CELL_LENGTH_Z])
{
        memset(cellList, -1, sizeof(cellList));
        for (int i = 0; i < NUM_PARTICLES; ++i) {
                int x = rand() % CELL_LENGTH_X;
                int y = rand() % CELL_LENGTH_Y;
                int z = rand() % CELL_LENGTH_Z;

                struct Particle particle = {
                        .particleID = i,
                        .x = x * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .y = y * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .z = z * CELL_CUTOFF_RADIUS + ((float) rand() / RAND_MAX) * CELL_CUTOFF_RADIUS,
                        .vx = 0,
                        .vy = 0,
                        .vz = 0,
                };
                memcpy(&cellList[x][y][z].particleList[cellList[x][y][z].next++], &particle, sizeof(struct Particle));
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
        struct Cell cellList[CELL_LENGTH_X][CELL_LENGTH_Y][CELL_LENGTH_Z] = {{{0}}};
        init(cellList);
        for (int i = 0; i < NUM_PARTICLES; ++i) {
                printf("particle %d: (%f, %f, %f)\n", i, particleList[i].x, particleList[i].y, particleList[i].z);
        }
        printf("\n");

        float accelerations[NUM_PARTICLES][3];

        // timestep
        for (int t = 0; t < TIME_STEPS; ++t) {
                memset(accelerations, 0, sizeof(accelerations));

                // force computation
                for (int x = 0; x < CELL_LENGTH_X; ++x) {
                        for (int y = 0; y < CELL_LENGTH_Y; ++y) {
                                for (int z = 0; z < CELL_LENGTH_Z; ++z) {
                                        struct Cell *homeCell = &cellList[x][y][z];
                                        struct Cell *neighborCells[13];

                                        int x_plus_1 = !(x == CELL_LENGTH_X - 1) * (x + 1);
                                        int y_plus_1 = !(y == CELL_LENGTH_Y - 1) * (y + 1);
                                        int z_plus_1 = !(z == CELL_LENGTH_Z - 1) * (z + 1);

                                        int y_minus_1 = !(y == 0) * (y - 1) + (y == 0) * (CELL_LENGTH_Y - 1);
                                        int z_minus_1 = !(z == 0) * (z - 1) + (z == 0) * (CELL_LENGTH_Z - 1);

                                        neighborCells[0] = &cellList[x_plus_1][y_minus_1][z_minus_1];
                                        neighborCells[1] = &cellList[x_plus_1][y_minus_1][z];
                                        neighborCells[2] = &cellList[x_plus_1][y_minus_1][z_plus_1];

                                        neighborCells[3] = &cellList[x_plus_1][y][z_minus_1];
                                        neighborCells[4] = &cellList[x_plus_1][y][z];
                                        neighborCells[5] = &cellList[x_plus_1][y][z_plus_1];

                                        neighborCells[6] = &cellList[x_plus_1][y_plus_1][z_minus_1];
                                        neighborCells[7] = &cellList[x_plus_1][y_plus_1][z];
                                        neighborCells[8] = &cellList[x_plus_1][y_plus_1][z_plus_1];

                                        neighborCells[9]  = &cellList[x][y_plus_1][z_minus_1];
                                        neighborCells[10] = &cellList[x][y_plus_1][z];
                                        neighborCells[11] = &cellList[x][y_plus_1][z_plus_1];

                                        neighborCells[12] = &cellList[x][y][z_plus_1];

                                        for (int n = 0; n < 13; ++n) {
                                                for (int i = 0; i < homeCell->next; ++i) {
                                                        for (int j = 0; j < neighborCells[n]->next; ++j) {
                                                                float ax = force_computation(homeCell->particleList[i].x, neighborCells[n]->particleList[j].x) * DT;
                                                                float ay = force_computation(homeCell->particleList[i].y, neighborCells[n]->particleList[j].y) * DT;
                                                                float az = force_computation(homeCell->particleList[i].z, neighborCells[n]->particleList[j].z) * DT;
                                                                int referenceParticleId = homeCell->particleList[i].particleID;
                                                                int neighborParticleId = neighborCells[n]->particleList[j].particleID;

                                                                accelerations[referenceParticleId][0] += ax;
                                                                accelerations[referenceParticleId][1] += ay;
                                                                accelerations[referenceParticleId][2] += az;
                                                                accelerations[neighborParticleId][0] -= ax;
                                                                accelerations[neighborParticleId][1] -= ay;
                                                                accelerations[neighborParticleId][2] -= az;
                                                        }
                                                }
                                        }
                                        for (int i = 0; i < homeCell->next; ++i) {
                                                for (int j = i + 1; j < homeCell->next; ++j) {
                                                        float ax = force_computation(homeCell->particleList[i].x, homeCell->particleList[i].x) * DT;
                                                        float ay = force_computation(homeCell->particleList[i].y, homeCell->particleList[i].y) * DT;
                                                        float az = force_computation(homeCell->particleList[i].z, homeCell->particleList[i].z) * DT;
                                                        int referenceParticleId = homeCell->particleList[i].particleID;
                                                        int neighborParticleId = homeCell->particleList[j].particleID;

                                                        accelerations[referenceParticleId][0] += ax;
                                                        accelerations[referenceParticleId][1] += ay;
                                                        accelerations[referenceParticleId][2] += az;
                                                        accelerations[neighborParticleId][0] -= ax;
                                                        accelerations[neighborParticleId][1] -= ay;
                                                        accelerations[neighborParticleId][2] -= az;
                                                }
                                        }
                                }
                        }
                }

                // TODO: motion update
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