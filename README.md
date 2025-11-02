# Soapy

## XI table

ok, so the xi_lmk seems very triangular in nature
so there are two options: laying it out as a flat array with wasted space
or doing some extra index magic

from what i understand the values are gonna be 
l in [0, l_max], m in [0, l], k in [m, l]

which means that the total size is

\sum_l \sum_m (l - m + 1)

(l_max + 1)(l_max + 2)(l_max + 3) / 6

# Dependencies

- `Vulkan` for dispatching and running GPU workloads
- `cnpy` for uploading numpy arrays
