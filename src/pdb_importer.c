#include "pdb_importer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/errno.h>

int import_atoms(char *filename, int **particle_ids, float **x, float **y, float **z, int *particle_count)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("fopen");
        return errno;
    }

    char line[80];
    int count = 0;

    int *local_particle_ids = NULL;
    float *local_x = NULL;
    float *local_y = NULL;
    float *local_z = NULL;

    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "ATOM", 4) != 0)
            continue;
        local_particle_ids = realloc(local_particle_ids, sizeof(float) * (count + 1));
        local_x = realloc(local_x, sizeof(float) * (count + 1));
        local_y = realloc(local_y, sizeof(float) * (count + 1));
        local_z = realloc(local_z, sizeof(float) * (count + 1));
        if (local_particle_ids == NULL || local_x == NULL || local_y == NULL || local_z == NULL) {
            perror("realloc");
            return errno;
        }

        char float_buffer[9] = {0};
        memcpy(float_buffer, line + 30, 8);
        local_x[count] = strtof(float_buffer, NULL);
        memcpy(float_buffer, line + 38, 8);
        local_y[count] = strtof(float_buffer, NULL);
        memcpy(float_buffer, line + 46, 8);
        local_z[count] = strtof(float_buffer, NULL);

        local_particle_ids[count] = count; //strtol(line + 6, NULL, 0);

        ++count;
    }

    fclose(file);

    *particle_count = count;
    *particle_ids = local_particle_ids;
    *x = local_x;
    *y = local_y;
    *z = local_z;

    return 0;
}

void create_cell_list(const int *particle_ids, const float *x, const float *y, const float *z,
                      int particle_count, struct Cell *cell_list, int cell_cutoff_radius,
                      int cell_dim_x, int cell_dim_y, int cell_dim_z)
{
    int *free_idx = (int *) calloc(cell_dim_x * cell_dim_y * cell_dim_z, sizeof(int));
    int cell_idx;

    for (int i = 0; i < particle_count; ++i) {
        int x_cell = x[i] / cell_cutoff_radius;
        int y_cell = y[i] / cell_cutoff_radius;
        int z_cell = z[i] / cell_cutoff_radius;
        if (x_cell >= 0 && x_cell < cell_dim_x && y_cell >= 0 && y_cell < cell_dim_y && z_cell >= 0 && z_cell < cell_dim_z) {
            cell_idx = x_cell + y_cell * cell_dim_x + z_cell * cell_dim_x * cell_dim_y;
            if (cell_idx >= 0 && cell_idx < cell_dim_x * cell_dim_y * cell_dim_z) {
                int free_slot = free_idx[cell_idx];
                if (free_slot < MAX_PARTICLES_PER_CELL) {
                    cell_list[cell_idx].particle_ids[free_slot] = particle_ids[i];

                    cell_list[cell_idx].x[free_slot] = x[i];
                    cell_list[cell_idx].y[free_slot] = y[i];
                    cell_list[cell_idx].z[free_slot] = z[i];

                    cell_list[cell_idx].vx[free_slot] = 0;
                    cell_list[cell_idx].vy[free_slot] = 0;
                    cell_list[cell_idx].vz[free_slot] = 0;

                    free_idx[cell_idx]++;
                } else {
                    printf("Warning: Cell %d is full, particle %d cannot be added\n", cell_idx, i);
                }
            } else {
                printf("Error: Computed cell_idx %d is out of bounds\n", cell_idx);
            }
        } else {
            printf("Error: Particle %d is out of bounds: (%.2f, %.2f, %.2f)\n", i, x[i], y[i], z[i]);
        }
    }

    for (int i = 0; i < cell_dim_x * cell_dim_y * cell_dim_z; ++i) {
        memset(&cell_list[i].particle_ids[free_idx[i]], -1, (MAX_PARTICLES_PER_CELL - free_idx[i]) * sizeof(int));
    }

    free(free_idx);
}

void cell_list_to_csv(struct Cell *cell_list, int num_cells, char *filename, const char *mode)
{
    FILE *file = fopen(filename, mode);

    for (int i = 0; i < num_cells; ++i) {
        int count = 0;
        struct Cell current_cell = cell_list[i];
        while (count < MAX_PARTICLES_PER_CELL && current_cell.particle_ids[count] != -1) {
            fprintf(file, "%d,%f,%f,%f\n", current_cell.particle_ids[count],
                                           current_cell.x[count],
                                           current_cell.y[count],
                                           current_cell.z[count]);
            count++;
        }
    }
    fprintf(file, "\n");

    fclose(file);
}
