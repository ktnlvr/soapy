#include <stdio.h>
#include <stdlib.h>

int read_xyz_coords(const char *filename,
                    double **x_out,
                    double **y_out,
                    double **z_out,
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

    double *x = malloc(n_atoms * sizeof(double));
    double *y = malloc(n_atoms * sizeof(double));
    double *z = malloc(n_atoms * sizeof(double));

    if (!x || !y || !z) {
        free(x);
        free(y);
        free(z);
        fclose(f);
        return 3;
    }

    char label[64];

    for (int i = 0; i < n_atoms; i++) {
        if (fscanf(f, "%63s %lf %lf %lf", label, &x[i], &y[i], &z[i]) != 4) {
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