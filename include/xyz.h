#include <stdio.h>
#include <stdlib.h>

int read_xyz_coords(const char *filename,
                    float **x_out,
                    float **y_out,
                    float **z_out,
                    int *n_atoms_out)
{
    FILE *f = fopen(filename, "r");
    if (!f)
        return 1;

    int n_atoms;

    if (fscanf(f, "%d", &n_atoms) != 1) {
        fclose(f);
        return 2;
    }

    float *x = (float*)malloc(n_atoms * sizeof(float));
    float *y = (float*)malloc(n_atoms * sizeof(float));
    float *z = (float*)malloc(n_atoms * sizeof(float));

    if (!x || !y || !z) {
        free(x);
        free(y);
        free(z);
        fclose(f);
        return 3;
    }

    char label[64];

    for (int i = 0; i < n_atoms; i++) {
        if (fscanf(f, "%63s %f %f %f", label, &x[i], &y[i], &z[i]) != 4) {
            free(x);
            free(y);
            free(z);
            fclose(f);
            return 4;
        }
    }

    fclose(f);

    *x_out = x;
    *y_out = y;
    *z_out = z;
    *n_atoms_out = n_atoms;

    return 0;
}