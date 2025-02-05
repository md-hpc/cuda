#ifndef PDB_IMPORTER_H
#define PDB_IMPORTER_H

struct Particle {
    int particle_id;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

/**
 * Reads atomic data from a .pdb file, extracting positions for particles.
 * Dynamically allocates memory for the particle list and returns the count.
 * 
 * @param filename The input file name.
 * @param particle_list Pointer to store the allocated array of particles.
 * @param particle_count Pointer to store the number of particles imported.
 * @return 0 on success, errno on failure.
 */
int import_atoms(const char *const filename, struct Particle **particle_list, int *particle_count);

#endif /* PDB_IMPORTER_H */