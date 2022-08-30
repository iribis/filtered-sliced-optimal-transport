# Filtered Sliced Optimal Transport

Source code of the Scalable multi-class sampling via filtered sliced optimal transport.

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

Toy example two color example:
===================

    ./FSOT -n 1024 -d 2 --method twoclass -o twoclass.dat


Toy example Progressive sample:
===================

    ./FSOT -n 1024 --nbproj 128 -d 2 --method progressive -o progressive.dat


Toy example Image stippling:
===================

    ./FSOT -n 1024 --nbproj 256 --method image -o image.dat

Toy example monochrome stippling:
===================

    ./FSOT -n 1024 --nbproj 256 --method stippling -o stippling.dat

