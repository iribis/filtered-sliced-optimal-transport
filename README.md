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

Generates 1 set of 1 sample per pixel (4096 in total) in a 64 by 64 toroidal tile in dimension 2, stored in test.dat using default parameters.
This sample-set can then be used in a renderer to produce blue noise error distribution.

This code is based on that of https://github.com/loispaulin/Sliced-Optimal-Transport-Sampling. We thank them for their work and recommend reading their article "Sliced Optimal Transport Sampling".

## Examples
Toy example Blue noise error distribution:
===================

    ./FSOT -n 4096 --step 2048 --tileSize 64 -d 2 --method BNED -o tile_BNED_64_64_1spp.dat


Toy example two color example:
===================

    ./FSOT -n 1024 --nbproj 256 -d 2 --method twoclass -o twoclass.dat
    python3 ../render_two_class_pointset.py ../results/twoclass.dat ../results/twoclass.png


Toy example Progressive sample:
===================

    ./FSOT -n 1024 --nbproj 128 -d 2 --method progressive --nbSubdiv 4 -o ../results/progressive.dat
    python3 ../render_progressive_pointset.py ../results/progressive.dat ../results/progressive.png 4


Toy example Image stippling:
===================

    ./FSOT -n 4096 -p 1000 --nbproj 512 --method image -o ../results/image.dat
    python3 ../render_color_img.py ../results/image.dat ../results/image.png 0.138348 0.477158 0.727171 1.0

Toy example monochrome stippling:
===================

    ./FSOT -n 8192 -p 500 --nbproj 256 --method stippling -i ../resources/elephants.png -o ../results/stippling.dat
    python3 ../render_stippling.py ../results/stippling.dat ../results/stippling.png


