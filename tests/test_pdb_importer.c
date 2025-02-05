#include "pdb_importer.h"
#include <stdio.h>
#include <assert.h>


int main(int argc, char **argv)
{
    assert(argc == 3);

    struct Particle *particle_list = NULL;
    int particle_count = 0;

    assert(import_atoms(argv[1], &particle_list, &particle_count) == 0);
    assert(particle_list != NULL);
    assert(particle_count == 103);

    FILE *expected = fopen(argv[2], "r");

    for (int i = 0; i < particle_count; ++i) {
        struct Particle particle;
        fscanf(expected, "%d\t%f\t%f\t%f\t%f\t%f\t%f\n", &particle.particle_id, &particle.x, &particle.y, &particle.z, &particle.vx, &particle.vy, &particle.vz);
        assert(particle.particle_id == particle_list[i].particle_id);
        assert(particle.x == particle_list[i].x);
        assert(particle.y == particle_list[i].y);
        assert(particle.z == particle_list[i].z);
        assert(particle.vx == particle_list[i].vx);
        assert(particle.vy == particle_list[i].vy);
        assert(particle.vz == particle_list[i].vz);
    }

    printf("[ \033[0;32mPASSED\033[0m ] test_pdb_importer\n");

    return 0;
}