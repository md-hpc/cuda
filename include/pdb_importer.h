#ifndef PDB_IMPORTER_H
#define PDB_IMPORTER_H

#define MAX_PARTICLES_PER_CELL 128


struct Cell {
    int particle_ids[MAX_PARTICLES_PER_CELL];
    float x[MAX_PARTICLES_PER_CELL];
    float y[MAX_PARTICLES_PER_CELL];
    float z[MAX_PARTICLES_PER_CELL];
    float vx[MAX_PARTICLES_PER_CELL];
    float vy[MAX_PARTICLES_PER_CELL];
    float vz[MAX_PARTICLES_PER_CELL];
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
int import_atoms(char *filename, int **particle_ids, float **x, float **y, float **z, int *particle_count);

// TODO: update
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
void create_cell_list(const int *particle_ids, const float *x, const float *y, const float *z,
                      int particle_count, struct Cell *cell_list, int cell_cutoff_radius,
                      int cell_dim_x, int cell_dim_y, int cell_dim_z);

void cell_list_to_csv(struct Cell *cell_list, int num_cells, char *filename);

#endif /* PDB_IMPORTER_H */
