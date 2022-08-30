# Weighted Sliced Optimal Transport

Source code of the Perceptual sample stratification via weighted sliced optimal transport.

Dependancies:
=============
 + OpenMP (`brew install libomp` on macOS)

Code compilation:
=================

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make


Code toy execution:
===================

    ./FSOT -n 4096 -d 2 --method BNED -o test.dat

Generates 1 set of 1 sample per pixel in a 64 by 64 tile in dimension 2, stored in test.dat using default parameters

This code is based on that of https://github.com/loispaulin/Sliced-Optimal-Transport-Sampling. We thank them for their work and recommend reading their article "Sliced Optimal Transport Sampling".
