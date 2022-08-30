//
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <cstdlib>
#include "../Math/VecX.h"
#include "../Tools/iopointset.h"
#include "../Transport/slicedOptimalTransportNCube.h"
#include "../Transport/progressive_sampler.h"
#include "../Transport/image_based_sot.h"
#include "../Transport/two_class_example.h"
#include "../Transport/stippling_one_color.h"


#define DIM 2

using namespace std;

void usage(const char **argv){
    cerr << argv[0] << " [-o <OutputFileName>] "
                       "[-n <nbPoints>] [-m <nbRealisations>] [-p <nbIteration>]"
                       "[--nbproj <nbDirectionPerStep>] [-s <seed>] [-d <dimension>]" 
                       "[--tileSize <imageSpaceSize>] [--method <method>]"<< endl;
}

void handleParameters(int argc,
                      const char** argv,
                      string& outPrefix,
                      int& nbIter,
                      int& m,
                      int& p,
                      int& seed,
                      int& nbPointsets,
                      int& dim,
                      int& tileSize,
                      bool& silent,
                      string& method,
                      int& nbSubdiv){
    int i = 1;
    while (i < argc){
        if (!strncmp(argv[i], "-o", 2)) {
            outPrefix = (argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "--method", 8)) {
            method = (argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "-n", 2)) {
            p = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "--nbproj", 8)) {
            m = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "--tileSize", 10)) {
            tileSize = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "--nbSubdiv", 10)) {
            nbSubdiv = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "-m", 2)) {
            nbPointsets = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "-p", 2)) {
            nbIter = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "-s", 2)) {
            seed = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "-d", 2)) {
            dim = atoi(argv[i+1]);
            ++i;
        } else if (!strncmp(argv[i], "-h", 2) || !strncmp(argv[i], "--help", 6)) {
            cerr << "Help: " << endl;
            cerr << "Option list:" << endl;
            cerr << "\t-o <OutputFileName> (optional): Specifies an output file in which points will be written."
                 << "If unset standard output will be used" << endl;
            cerr << "\t-n <nbPoints> (default 1024): Specifies the number of points to generate" << endl;
            cerr << "\t-m <nbRealisations> (default 1): Specifies the number of generated pointsets" << endl;
            cerr << "\t-p <nbIteration> (default 4096): Specifies the number of batches in the optimization process" << endl;
            cerr << "\t--nbproj <nbDirectionPerStep> (default 64): Specifies the number of slices per batch in the optimization process" << endl;
            cerr << "\t--tileSize <imageSpaceSize> (default 64): Specifies the tile size" << endl;
            cerr << "\t-s <seed> (default 133742): Specifies the random seed" << endl;
            cerr << "\t-d <dimension> (default 2): Specifies samples dimension" << endl;
            cerr << "\t--method <sampling method> (default BNED): BNED (Blue Noise Error Distribution), progressive, image, stippling, twoclass or SOT" << endl;
            cerr << "\t--silent (optional): Cancels all outputs other than the points and errors" << endl;
            cerr << "\t" << endl;
            usage(argv);
            exit(2);
        } else if (!strncmp(argv[i], "--silent", 8)) {
            silent = true;
        } else {
            cerr << "Unknown option " << argv[i] << endl;
            exit(1);
        }
        ++i;
    }
}

template <class VECTYPE>
int main_template(int argc, const char **argv) {

    int nbIter = 4096;
    int dim = DIM;
    int tileSize = 64;
    int p = 4096;
    int m = 64;
    int nbPointsets = 1;
    int nbSubdiv = 1;
    //Default parameters value
    string outPrefix = "";
    string method = "BNED";
    int seed = 133742;
    bool silent = false;

    handleParameters(argc, argv, outPrefix, nbIter, m, p, seed, nbPointsets, dim, tileSize, silent, method, nbSubdiv);
    if((p/(tileSize*tileSize) != p/double(tileSize*tileSize)) && method == "BNED"){
        std::cout << "number of sample ("<< p <<") is not matching with the tile size (" << tileSize << ")"<< std::endl;
        return 0;
    }
    //If file name ends in .bin then the output will be written in binary
    ostream* out = &cout;
    if (outPrefix != "") {
        out = new ofstream(outPrefix);
        if (out->fail()) {
            cerr << "Can't open output file \"" + outPrefix + "\"" << endl;
            exit(3);
        }
        cerr << "Output file: " + outPrefix + ".dat" << endl;
    }

    mt19937 generator(seed);
    if (!silent) {
        cerr << "Generating " << nbPointsets << " sets of " << p << " points in " << dim << "D using " << nbIter << " batches of " << m
             << " slices" << endl;
    }
    
    for (int indPointset = 0; indPointset < nbPointsets; ++indPointset) {
        vector<VECTYPE> points(p, VECTYPE(dim));

        //Init from whitenoise in ball
        uniform_real_distribution<> unif(0., 1.0);
        normal_distribution<> norm(0., 1.0);

        for (size_t i = 0; i < points.size(); ++i) {
            points[i] = randomVectorInCube<VECTYPE>(dim, generator);
        }

        vector<VECTYPE> result;
            if(method == "SOT"){
                slicedOptimalTransportNCube(points, result, nbIter, m, seed, 1, silent);
            }if(method == "BNED"){
                slicedOptimalTransportNCube(points, result, nbIter, m, seed, tileSize, silent);
            }else if(method == "progressive"){
                slicedOptimalTransportNCube_progressive(points, result, nbIter, m, seed, tileSize, silent, nbSubdiv);
            }else if(method == "image"){
                slicedOptimalTransportNImageBased(points, result, nbIter, m, seed, silent);
            }else if(method == "twoclass"){
                slicedOptimalTransportNCube_two_class(points, result, nbIter, m, seed, tileSize, silent, nbSubdiv);
            }else if(method == "stippling"){
                slicedOptimalTransportNStippling(points, result, nbIter, m, seed, silent); 
            }else{
                std::cout << "Error: not findind the method" << std::endl;
            }
            

            if (indPointset != 0)
                *out << "#" << endl;

            savePointsetND(*out, result);
            for (size_t i = 0; i < result.size();++i){
                points[i] = result[i];
            }
        
    }
    return 0;
}

#include "../Tools/dimensionsInstantiation.hpp"
