#ifndef PDB_IMPORTER_H
#define PDB_IMPORTER_H

#define MAX_PARTICLES_PER_CELL 128


struct Particle {
    int particle_id;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

struct Cell {
    struct Particle particle_list[MAX_PARTICLES_PER_CELL];
};

// TODO: update
/**
 * Reads atomic data from a .pdb file, extracting positions for particles.
 * Dynamically allocates memory for the particle list and returns the count.
 * 
 * @param filename The input file name.
 * @param particle_list Pointer to store the allocated array of particles.
 * @param particle_count Pointer to store the number of particles imported.
 * @return 0 on success, errno on failure.
 */
int import_atoms(char *filename, float *particle_id, float *x, float *y, float *z, int *particle_count);

/**
 * @brief Assigns particles to a spatial cell grid based on their positions.
 *
 * This function distributes particles into a grid of cells for efficient spatial
 * partitioning.
 *
 * @param particle_list      Pointer to an array of particles.
 * @param particle_count     Number of particles in the particle_list.
 * @param cell_list          Pointer to an array of cells that store particles.
 * @param cell_cutoff_radius The size of each cell, defining the partitioning scale.
 *
 * @note This function assumes that the `particle_list` contains valid particle data
 *       with defined x, y, and z coordinates. Additionally, the `cell_list` must
 *       have sufficient memory allocated to store the particles.
 */
void create_cell_list(struct Particle *particle_list, int particle_count,
                      struct Cell *cell_list, int cell_cutoff_radius);

void cell_list_to_csv(struct Cell *cell_list, int num_cells, char *filename);

#endif /* PDB_IMPORTER_H */
