#include "pdb_importer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/errno.h>

int import_atoms(char *filename, struct Particle **particle_list, int *particle_count)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("fopen");
        return errno;
    }

    char line[80];
    struct Particle *particles = NULL;
    int count = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "ATOM", 4) != 0)
            continue;
        
        particles = realloc(particles, sizeof(struct Particle) * (count + 1));
        if (particles == NULL) {
            perror("realloc");
            return errno;
        }

        char float_buffer[9] = {0};
        memcpy(float_buffer, line + 30, 8);
        particles[count].x = strtof(float_buffer, NULL);
        memcpy(float_buffer, line + 38, 8);
        particles[count].y = strtof(float_buffer, NULL);
        memcpy(float_buffer, line + 46, 8);
        particles[count].z = strtof(float_buffer, NULL);

        particles[count].vx = 0;
        particles[count].vy = 0;
        particles[count].vz = 0;

        particles[count].particle_id = strtol(line + 6, NULL, 0);

        ++count;
    }

    fclose(file);

    *particle_list = particles;
    *particle_count = count;

    return 0;
}

void create_cell_list(struct Particle *particle_list, int particle_count,
                      struct Cell *cell_list, int cell_cutoff_radius)
{
    int free_idx[CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z] = {0};

    for (int i = 0; i < particle_count; ++i) {
        int x_cell = particle_list[i].x / cell_cutoff_radius;
        int y_cell = particle_list[i].y / cell_cutoff_radius;
        int z_cell = particle_list[i].z / cell_cutoff_radius;

        int cell_idx = x_cell + y_cell * CELL_LENGTH_X + z_cell * CELL_LENGTH_X * CELL_LENGTH_Y;

        cell_list[cell_idx].particle_list[free_idx[cell_idx]++] = particle_list[i];
    }
}

void cell_list_to_csv(struct Cell *cell_list, int num_cells, char *filename)
{
    FILE *file = fopen(filename, "w");
    fprintf(file, "particle_id,x,y,z\n");

    for (int i = 0; i < num_cells; ++i) {
        for (int j = 0; j < MAX_PARTICLES_PER_CELL; ++j) {
            fprintf(file, "%d,%f,%f,%f\n", cell_list[i].particle_list[j].particle_id, cell_list[i].particle_list[j].x, cell_list[i].particle_list[j].y, cell_list[i].particle_list[j].z);
        }
    }

    fclose(file);
}