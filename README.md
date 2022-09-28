# Filtered Sliced Optimal Transport

Source code of the Scalable multi-class sampling via filtered sliced optimal transport. This example is for demonstration purposes and has not been optimized. It is fully CPU based with a parallelization via OpenMP.

In case of problems or bugs don't hesitate to contact me I will do my best to solve it.

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

Several code examples are available to easily test the different applications (Blue noise error distribution, simple multi-class, progressive sampler and stippling). For each application we give an example of use with the command line to run the code and if necessary the python script to generate an understandable image of the result.

Toy example Blue noise error distribution:
===================

    ./FSOT -n 4096 --nbproj 2048 --tileSize 64 -d 2 --method BNED -o ../results/tile_BNED_64_64_1spp.dat

For the BNED method the result is in the form of a list of points. For each pixel the n samples are put in a sequence and the pixels are also put consecutively. By example, the sample m of the pixel (i,j) is located at the line (i*tile_size+j)*n+m.

Also create a .h file with the same name and location containing the corresponding tile and a method to ask the samples for a C++ renderer.

For this problem, we recmmand using a number of projections (--nbproj) proportional to the of pixels in the tile (tileSize*tileSize). We also recommand using between 4 000 and 20 000 iterations (-p).

Toy example two color example:
===================

    ./FSOT -n 1024 --nbproj 256 -d 2 --method twoclass -o twoclass.dat
    python3 ../render_two_class_pointset.py ../results/twoclass.dat ../results/twoclass.png


Toy example Progressive sample:
===================

    ./FSOT -n 1024 --nbproj 128 -d 2 --method progressive --nbSubdiv 4 -o ../results/progressive.dat
    python3 ../render_progressive_pointset.py ../results/progressive.dat ../results/progressive.png 4

To have a more visual result the subdivisions are not made on dyadic numbers. The results will be slightly different from those presented in the article.
The parameter (--nbSubdiv) control the number of subdivisions for wich the pointset is optimized.

Toy example Image stippling:
===================

    ./FSOT -n 4096 -p 1000 --nbproj 512 --method image -i ../resources/colored_sea.png -o ../results/image.dat
    python3 ../render_color_img.py ../results/image.dat ../results/image.png ../resources/colored_sea.png 0.133337 0.470049 0.719257 1.0

The python file generating the result takes as parameters the relative energies of the different channels of the cmyk image (4 numbers after the export file). They have been hard coded here to simplify but you can find them writen when luching the optimization. 
As guidance, we recommend using beetween 1 000 and 10 000 iterations (-p) and around 500 projections per iterations (--nbproj).

Toy example monochrome stippling:
===================

    ./FSOT -n 8192 -p 500 --nbproj 256 --method stippling -i ../resources/elephants.png -o ../results/stippling.dat
    python3 ../render_stippling.py ../results/stippling.dat ../results/stippling.png ../resources/elephants.png


## Parameters

See help for the parameters:
===================

    ./FSOT -h

The main parameters are (-n) the number of points, (-p) the number of iterations, (--nbproj) the number of projections and (-o) for the output file. The reste of the parameters are not used in every methods.
