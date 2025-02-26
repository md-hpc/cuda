#include "pdb_importer.h"
#include <stdio.h>
#include <assert.h>

void naive(struct Particle *particle_list, const int particle_count);

int main(int argc, char **argv)
{
        assert(argc == 3);

        // import pdb data
        struct Particle *particle_list = NULL;
        int particle_count = 0;
        assert(import_atoms(argv[1], &particle_list, &particle_count) == 0);

        // run 
        naive(particle_list, particle_count);

        // output data
	FILE *file = fopen(argv[2], "r");
        fprintf(file, "particle_id,x,y,z\n");
        for (int i = 0; i < particle_count; ++i) {
                fprintf(file, "%d,%f,%f,%f\n", particle_list[i].particle_id, particle_list[i].x, particle_list[i].y, particle_list[i].z);
        }

        return 0;
}
