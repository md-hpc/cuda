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
    int cell_idx;

    for (int i = 0; i < particle_count; ++i) {
        int x_cell = particle_list[i].x / cell_cutoff_radius;
        int y_cell = particle_list[i].y / cell_cutoff_radius;
        int z_cell = particle_list[i].z / cell_cutoff_radius;
	if (x_cell >= 0 && x_cell < CELL_LENGTH_X && y_cell >= 0 && y_cell < CELL_LENGTH_Y && z_cell >= 0 && z_cell < CELL_LENGTH_Z) {
		cell_idx = x_cell + y_cell * CELL_LENGTH_X + z_cell * CELL_LENGTH_X * CELL_LENGTH_Y;
		if (cell_idx >= 0 && cell_idx < CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z) {
			if (free_idx[cell_idx] < MAX_PARTICLES_PER_CELL) {
				cell_list[cell_idx].particle_list[free_idx[cell_idx]++] = particle_list[i];
			} else {
				printf("Warning: Cell %d is full, particle %d cannot be added\n", cell_idx, i);
										            }
		} else {
			printf("Error: Computed cell_idx %d is out of bounds\n", cell_idx);
							    }
	} else {
		printf("Error: Particle %d is out of bounds: (%.2f, %.2f, %.2f)\n", i, particle_list[i].x, particle_list[i].y, particle_list[i].z);
	}



        cell_list[cell_idx].particle_list[free_idx[cell_idx]++] = particle_list[i];
    }

    for (int i = 0; i < CELL_LENGTH_X * CELL_LENGTH_Y * CELL_LENGTH_Z; ++i) {
        memset(&cell_list[i].particle_list[free_idx[i]], -1, (MAX_PARTICLES_PER_CELL - free_idx[i]) * sizeof(struct Particle));
    }
}

void cell_list_to_csv(struct Cell *cell_list, int num_cells, char *filename)
{
    FILE *file = fopen(filename, "w");
    fprintf(file, "particle_id,x,y,z\n");

    for (int i = 0; i < num_cells; ++i) {
        int count = 0;
        struct Cell current_cell = cell_list[i];
        while (current_cell.particle_list[count].particle_id != -1) {
            fprintf(file, "%d,%d,%f,%f,%f\n", i,
                                             current_cell.particle_list[count].particle_id,
                                             current_cell.particle_list[count].x,
                                             current_cell.particle_list[count].y,
                                             current_cell.particle_list[count].z);
            count++;
        }
    }

    fclose(file);
}
