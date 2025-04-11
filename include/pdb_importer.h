#ifndef PDB_IMPORTER_H
#define PDB_IMPORTER_H

#ifndef MAX_PARTICLES_PER_CELL
#define MAX_PARTICLES_PER_CELL 128
#endif


struct Cell {
    int particle_ids[MAX_PARTICLES_PER_CELL];
    float x[MAX_PARTICLES_PER_CELL];
    float y[MAX_PARTICLES_PER_CELL];
    float z[MAX_PARTICLES_PER_CELL];
    float vx[MAX_PARTICLES_PER_CELL];
    float vy[MAX_PARTICLES_PER_CELL];
    float vz[MAX_PARTICLES_PER_CELL];
};

/**
 * Reads atomic data from a .pdb file, extracting positions for particles.
 * Dynamically allocates memory for the particle list and returns the count.
 * 
 * @param[in] filename         Path to the input file containing ATOM lines.
 * @param[out] particle_ids    Pointer to an array of particle IDs (allocated inside the function).
 * @param[out] x               Pointer to an array of x-coordinates (allocated inside the function).
 * @param[out] y               Pointer to an array of y-coordinates (allocated inside the function).
 * @param[out] z               Pointer to an array of z-coordinates (allocated inside the function).
 * @param[out] particle_count  Pointer to an integer where the number of particles will be stored.
 *
 * @return 0 on success, or a non-zero error code (errno) if an error occurs during file I/O or memory allocation.
 *
 * @note The caller is responsible for freeing the memory allocated for `particle_ids`, `x`, `y`, and `z`.
 */
int import_atoms(char *filename, int **particle_ids, float **x, float **y, float **z, int *particle_count);

/**
 * @brief Populates a 3D spatial cell list with particle data based on their positions.
 *
 * This function assigns particles to spatial grid cells based on their coordinates
 * and a given cutoff radius. Each cell can hold up to `MAX_PARTICLES_PER_CELL` particles.
 * The function initializes velocity components to zero and fills remaining particle slots 
 * in each cell with -1 to mark them as unused.
 *
 * @param[in] particle_ids       Array of particle IDs.
 * @param[in] x                  Array of x-coordinates of particles.
 * @param[in] y                  Array of y-coordinates of particles.
 * @param[in] z                  Array of z-coordinates of particles.
 * @param[in] particle_count     Total number of particles.
 * @param[out] cell_list         Array of cells representing the spatial grid. 
 *                               Must be preallocated with size `cell_dim_x * cell_dim_y * cell_dim_z`.
 * @param[in] cell_cutoff_radius The cutoff radius used to determine cell dimensions.
 * @param[in] cell_dim_x         Number of cells along the x-axis.
 * @param[in] cell_dim_y         Number of cells along the y-axis.
 * @param[in] cell_dim_z         Number of cells along the z-axis.
 *
 * @note The function assumes the cell list has been preallocated.
 *       Each `struct Cell` should contain arrays of size `MAX_PARTICLES_PER_CELL`
 *       for particle IDs and positions/velocities.
 */
void create_cell_list(const int *particle_ids, const float *x, const float *y, const float *z,
                      int particle_count, struct Cell *cell_list, int cell_cutoff_radius,
                      int cell_dim_x, int cell_dim_y, int cell_dim_z);

/**
 * @brief Writes particle data from a cell list to a CSV file.
 *
 * This function iterates over a list of spatial cells and writes each valid particle's 
 * ID and position (x, y, z) to a CSV file. Cells are assumed to contain particles
 * indexed up to `MAX_PARTICLES_PER_CELL`, with `-1` marking unused entries.
 *
 * @param[in] cell_list   Array of `struct Cell` containing particle data.
 * @param[in] num_cells   Total number of cells in the `cell_list` array.
 * @param[in] filename    Name of the CSV file to write to.
 * @param[in] mode        File mode string (e.g., `"w"` to overwrite, `"a"` to append).
 *
 * @note Each line in the CSV file corresponds to a single particle and contains:
 *       particle ID, x, y, z. An empty line is printed after all particles.
 *
 * @warning If the file cannot be opened, behavior is undefined (no error checking is performed).
 */
void cell_list_to_csv(struct Cell *cell_list, int num_cells, char *filename, const char *mode);

#endif /* PDB_IMPORTER_H */
