#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/errno.h>

struct Particle {
    int particle_id;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

int import_atoms(const char *const filename, struct Particle **particle_list, int *particle_count)
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
