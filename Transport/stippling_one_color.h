//
//

#ifndef stippling_SLICEDOPTIMALTRANSPORT_H
#define stippling_SLICEDOPTIMALTRANSPORT_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include "../Math/VecX.h"
#include "../Math/myMath.h"
#include "../Tools/iopointset.h"
#include "../Tools/mapping.h"
#include "../Tools/my_utility.h"
#include <fstream> // ifstream
#include <sstream> // stringstream
#include <string>

#include "../Tools/lodepng.h"

const int multiplier_pointset_images_stippling = 5;

int w_stippling = 10000;
int h_stippling = 10000;

std::vector<std::vector<std::vector<float>>> image_stippling;
std::vector<std::vector<std::vector<float>>> image_stippling_CDF;
std::vector<std::vector<float>> cumulativeimage_stippling;
std::vector<float> sum_stippling;

void decodeOneStep(const char* filename) {
  int row = 0, col = 0, numrows = 0, numcols = 0, max_val = 256;
  std::vector<unsigned char> image; //the raw pixels
  unsigned width, height;

  //decode
  unsigned error = lodepng::decode(image, width, height, filename);

  //if there's an error, display it
  if(error){
    std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
  }else{
    std::cout << "Image loaded : w(" << width <<"), h("<< height << ")" <<std::endl;
  }
  w_stippling = width;
  h_stippling = height;
  numrows = width;
  numcols = height;
  sum_stippling = std::vector<float>(2);
  cumulativeimage_stippling = std::vector<std::vector<float>>(width);
  image_stippling_CDF = std::vector<std::vector<std::vector<float>>>(width);
  image_stippling = std::vector<std::vector<std::vector<float>>>(width);
  for (size_t j = 0; j < width; j++)
  {
    cumulativeimage_stippling[j] = std::vector<float>(2);
  }
  for (size_t i = 0; i < width; i++){
        image_stippling_CDF[i] = std::vector<std::vector<float>>(height);
        image_stippling[i] = std::vector<std::vector<float>>(height);
    for (size_t j = 0; j < height; j++){
        image_stippling_CDF[i][j] = std::vector<float>(2);
        image_stippling[i][j] = std::vector<float>(2);
    }
  }
  for(row = 0; row < width; ++row){
    for (col = 0; col < height; ++col){
        image_stippling[row][col][0] = int(image[4*(col*width+row)]);
        image_stippling[row][col][1] = int(image[4*(col*width+row)]);
    }
  }
  for (int c = 0; c < 2; ++c){
    // Now print the array to see the result
    float cumulat = 0;
    for(row = 0; row < numrows; ++row){
        cumulat = 0;
        for (col = 0; col < numcols; ++col){
            image_stippling[row][col][c] = 255 - int((image_stippling[row][col][c]/float(max_val))*255);
            image_stippling_CDF[row][col][c] = cumulat + image_stippling[row][col][c];
            cumulat += image_stippling[row][col][c];
        }

    }
    sum_stippling[c] = 0;
    cumulat = 0;
    for(row = 0; row < numrows; ++row){
        float cumulative = 0;
        for (col = 0; col < numcols; ++col){
            cumulative += image_stippling[row][col][c];
        }
        cumulativeimage_stippling[row][c] = cumulat + cumulative;
        cumulat += cumulative;
        sum_stippling[c] = cumulativeimage_stippling[row][c];
    }
  }
  std::cout << "end loading" << std::endl;
  //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
}

void readimage_stippling(const char* filename){
    decodeOneStep(filename);

}


template <class VECTYPE>
void project_stippling(const std::vector<VECTYPE>& points, std::vector<std::pair<double, int>> &pointsProject, const VECTYPE& dir,int N, VECTYPE& offset, int selector){
    for(size_t s = 0; s < points.size(); ++s){
            VECTYPE p = points[s];
            //double proj = (toroidal_minus(p, offset) * dir);
            double proj = (p * dir);
            std::pair<double, int> val_indice = std::pair<double, int>(proj, s);
            pointsProject.push_back(val_indice);

    }

}


template <class VECTYPE>
inline void getInverseimage_stippling(int D, int nbSamples, std::vector<double>& pos,VECTYPE dir,VECTYPE offset, int selector){

    pos.resize(nbSamples);
    std::vector<double> posbis;
    posbis.resize(nbSamples*multiplier_pointset_images_stippling);
    for (int i = 0; i < nbSamples*multiplier_pointset_images_stippling; ++i){
        int img = 0;
        VECTYPE p;
        float rnd1 = ((float)rand() / RAND_MAX)*sum_stippling[img];

        float x = 0;
        float y = 0;

        int low = 0;
        int up = w_stippling-1;
        int mid = (low+up)/2;
        while(up-low > 1){
            mid = (low+up)/2;
            if(cumulativeimage_stippling[mid][img]<=rnd1){
                low = mid;
            }else{
                up = mid;
            }
        }
        if(up-low == 0){
            x = low;
        }else if(cumulativeimage_stippling[low][img]<=rnd1){
            x = low;
        }else{
            x = up;
        }

        
        float rnd2 = ((float)rand() / RAND_MAX)*image_stippling_CDF[int(x)][h_stippling-1][img];

        low = 0;
        up = h_stippling-1;
        mid = (low+up)/2;
        while(up-low > 1){
            mid = (low+up)/2;
            if(image_stippling_CDF[int(x)][mid][img]<=rnd2){
                low = mid;
            }else{
                up = mid;
            }
        }
        if(up-low == 0){
            y = low;
        }else if(image_stippling[int(x)][low][img]<=rnd2){
            y = low;
        }else{
            y = up;
        }

        p[0] = ((x+((float)rand() / RAND_MAX))/w_stippling)*w_stippling/std::max(w_stippling,h_stippling);
        p[1] = (1-(y+((float)rand() / RAND_MAX))/h_stippling)*h_stippling/std::max(w_stippling,h_stippling);
        posbis[i] = p*dir;
    }


    std::sort(posbis.begin(),posbis.end());
    for (int i = 0; i < nbSamples; i++)
    {
        pos[i] = posbis[i*multiplier_pointset_images_stippling + int(multiplier_pointset_images_stippling/2)];
    }
    
}

/**
 * Compute optimal transport in 1D for direction \f$ \theta \f$ and \f$d_{j}\f$ being the 1D displacement of \f$\x^j\f$
 * that minimize the 1D sliced optimal transport along \f$ \theta \f$.
 *
 * Denoting $\sigma$ the permutations of the indices \f$\{j\}_{j=1..N}\f$ such that
 * \f$\bigl(\x^{\sigma(j)} \cdot \theta \bigr)_j\f$, is a sorted sequence of increasing values,
 * one can compute \f$d_{j}\f$ via
 * \f$ d_{j} = C_{\theta}^{-1}\left(\frac{\sigma(j)-\frac12}{N}\right)\,. \vspace*{-1mm}\f$
 *
 * @param dir Direction \f$ \theta \f$
 * @param points Table containing the points \f$ x_j \f$
 * @param pos Table containing the optimal solution in 1D
 * @param shift Output the 1D shift to apply to minimize transport cost
 * @param pointsProject Memory buffer used to store the 1D projection of points. Must be the same size as \p points
 */
template<class VECTYPE>
inline void slicedStepNStippling(const VECTYPE& dir,
                            const std::vector<VECTYPE>& points,
                            std::vector<double>& shift)
{
    int N = points.front().dim();
    for (size_t i = 0; i < points.size(); ++i){
        shift[i] = 0.0;
    }
    VECTYPE offset;
    for (int i = 0; i < N; i++)
    {
        offset[i] =0*((float)rand() / RAND_MAX);
    }
    int selector = rand() % 15;//12
    std::vector<std::pair<double, int>> pointsProject;
    project_stippling(points, pointsProject, dir, N,offset,selector);
    std::vector<double> pos(pointsProject.size());
    getInverseimage_stippling(N,pointsProject.size(), pos,dir,offset,selector);
    std::sort(pointsProject.begin(), pointsProject.end());
    double normalise_factor = 1.0;
    double grad_normalization = float(pointsProject.size())/points.size();
    //Computes required shift to optimize 1D optimal transport
    for (size_t i = 1; i < pointsProject.size()-1; ++i) {
        //Compute shifting
        double s = pos[i] - pointsProject[i].first;
        normalise_factor = ((pos[i + 1] - pos[i - 1])/2.0)* pointsProject.size();
        shift[pointsProject[i].second] += grad_normalization*(s/normalise_factor);
    }
    double s = pos[0] - pointsProject[0].first;
    normalise_factor = (pos[1] - pos[0]) * pointsProject.size();
    shift[pointsProject[0].second] += grad_normalization*(s / normalise_factor);
    s = pos[pointsProject.size() - 1] - pointsProject[pointsProject.size() - 1].first;
    normalise_factor = (pos[pointsProject.size() - 1] - pos[pointsProject.size() - 1 - 1]) * pointsProject.size();
    shift[pointsProject[pointsProject.size() - 1].second] += grad_normalization*(s / normalise_factor);
}

/**
 * Compute optimal transport in 1D for the \p directions and displace \f$ x_j \f$ by
 * \f$\pmb{\delta}^j \EqDef \frac{1}{K} \sum_{i=1}^K d_{i,j}\, \theta_i \vspace*{-1mm}\f$ with
 * with \f$ d_{i,j} \f$ being the displacement of \f$\x^j\f$ that minimize the 1D sliced optimal
 * transport along direction \f$ \theta_i \f$
 *
 * @param pointsOut Table containing the points \f$ x_j \f$
 * @param directions Table containing the \f$\theta_i\f$
 * @param pos Table containing the 1D positions optimizing transport in 1D
 * @param shift Used to avoid having to allocate uge chunks of memory. Must be a vector of size m containing vectors of same size as \p pointsOut.
 * @param finalShift Used to avoid having to allocate huge chunks of memory Must be a vector of same size as \p pointsOut.
 * @param pointsProject Used to avoid having to allocate uge chunks of memory. Must be a vector of size m containing vectors of same size as \p pointsOut.
 * @return the Wasserstein cost of the current iteration.
 */
template <class VECTYPE>
inline void slicedOptimalTransportBatch_stippling(std::vector<VECTYPE>& pointsOut,
                                 const std::vector<VECTYPE>& directions,
                                 std::vector<std::vector<double>>& shift,
                                 std::vector<VECTYPE>& finalShift)
 {

    int m = directions.size();
    int nbPoints = pointsOut.size();

    //Compute the shift along each direction
#pragma omp parallel for shared(directions, shift)
    for (int k = 0; k < m; ++k){
        for(double& v : shift[k]){
			v = 0.;
        }
        const VECTYPE& dir = directions[k];

        slicedStepNStippling(dir, pointsOut, shift[k]);
    }

        //Accumulate shift from all directions
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i) {
        VECTYPE sh(finalShift[i].dim());
        memset(&sh[0], 0, finalShift[i].dim() * sizeof(sh[0]));
        for (int k = 0; k < m; ++k) {
            sh += shift[k][i] * directions[k];
        }
        finalShift[i] = sh;
        finalShift[i] /= m;
    }

    //Displace points according to accumulated shift
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i) {
        pointsOut[i] += finalShift[i];
    }

}

void print_progress_stippling(double ratio){
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * ratio;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << (ratio * 100.0) << " %\r";
    std::cout.flush();
}

/**
 *  \brief Computes an optimized point set to uniformly sample the unit N-Ball using sliced optimal transport
 *
 * @param pointsIn Contains input ND points
 * @param pointsOut Contains optimized ND points
 * @param nbIter Number of iterations
 * @param m Number of slice per iteration
 * @param seed random seed
 * @return the Sliced Wasserstein distance between the samples and the uniform distribution.
 */

template <class VECTYPE>
inline void slicedOptimalTransportNStippling(const std::vector<VECTYPE>& pointsIn,
                                        std::vector<VECTYPE>& pointsOut,
                                        int nbIter,
                                        int m,
                                        int seed, const char* inPrefix, bool silent)
{

    int N = pointsIn.front().dim();
    pointsOut = pointsIn;
    readimage_stippling(inPrefix);
    //Accumulation shift to be applied later on
    std::vector<std::vector<double>> shift(m, std::vector<double>(pointsOut.size()));
    std::vector<VECTYPE> finalShift(pointsOut.size(), VECTYPE(N));

    std::vector<VECTYPE> directions(m, VECTYPE(N));

    //Iterate 1D Optimal Transport
    for (int i = 0; i < nbIter; i += 1){
        if(!silent){
            print_progress_stippling(double(i)/nbIter);
        }
        chooseDirectionsND(directions, m, seed);

        slicedOptimalTransportBatch_stippling(pointsOut, directions, shift, finalShift);
    }
    print_progress(1.0);

}

#endif
