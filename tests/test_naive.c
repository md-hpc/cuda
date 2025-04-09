#include "pdb_importer.h"
#include <stdio.h>
#include <assert.h>

void naive(int *particle_ids, float *x, float *y, float *z, int particle_count);

int main(int argc, char **argv)
{
        assert(argc == 3);

        // import pdb data
        int *particle_ids;
        float *x;
        float *y;
        float *z;
        int particle_count;
        assert(import_atoms(argv[1], &particle_ids, &x, &y, &z, &particle_count) == 0);

        // run 
        naive(particle_ids, x, y, z, particle_count);

        // output data
	FILE *file = fopen(argv[2], "w");
        fprintf(file, "particle_id,x,y,z\n");
        for (int i = 0; i < particle_count; ++i) {
                fprintf(file, "%d,%f,%f,%f\n", particle_ids[i], x[i], y[i], z[i]);
        }

        return 0;
}
