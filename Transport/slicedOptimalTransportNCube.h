//
//

#ifndef SLICEDOPTIM_NCUBESLICEDOPTIMALTRANSPORT_H
#define SLICEDOPTIM_NCUBESLICEDOPTIMALTRANSPORT_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include "../Math/VecX.h"
#include "../Math/myMath.h"
#include "../Tools/iopointset.h"
#include "../Tools/my_utility.h"

int tileSize = 64;
double sigma = 2.1;
int kernel_size = 4;
int reconstruction_kernel_size = 4;
double mul = 3;
double r = 2;
double alpha = 2;


double w_recons(float di, float dj) {
    return std::max(0.0,std::exp(-alpha * di * di) - std::exp(-alpha * r * r)) * std::max(0.0,std::exp(-alpha * dj * dj) - std::exp(-alpha * r * r));
}

template <class VECTYPE>
double weight(VECTYPE& c, VECTYPE& p){
	
    double res = 0;
    for(int di = -reconstruction_kernel_size+1; di < reconstruction_kernel_size; ++di){
        for(int dj = -reconstruction_kernel_size+1; dj < reconstruction_kernel_size; ++dj){
            double a = std::min(std::min(std::abs(c[0] + di - p[0]),std::abs(c[0] + di - p[0] + tileSize)),std::abs(c[0] + di - p[0] - tileSize));
            double b = std::min(std::min(std::abs(c[1] + dj - p[1]),std::abs(c[1] + dj - p[1] + tileSize)),std::abs(c[1] + dj - p[1] - tileSize));
            res += exp(-(di*di+dj*dj)/pow(sigma, 2.0)) * w_recons(a,b);
        }
    }
    return res;
}

template <class VECTYPE>
void project(const std::vector<VECTYPE>& points,
            std::vector<std::pair<double, int>>& pointsProject,
            const VECTYPE& dir, VECTYPE& center, int functionSelection,
            double w_ref, int dim){

    int spp = points.size()/(tileSize*tileSize);

    // Weighting Function Selection
    int k = functionSelection;
    int u = k/int(tileSize);
    int v = k%int(tileSize);
    
    if(tileSize == 1){
        kernel_size = 1;
    }
    for(int di = -kernel_size+1; di < kernel_size; ++di){
        for(int dj = -kernel_size+1; dj < kernel_size; ++dj){
            int x = ((u+di + tileSize)%tileSize);
            int y = ((v+dj + tileSize)%tileSize);
            int pixel_indice =  (x*tileSize + y)*spp;
            VECTYPE c;
            c[0] = u + 0.5;
            c[1] = v + 0.5;
            for(int s = 0; s < spp; ++s){
                VECTYPE p = points[pixel_indice+s];
                //if (weight(c,p) >= w_ref) {
                    double proj = toroidal_minus(p,center) * dir;

                    int indices = pixel_indice+s;
                    std::pair<double, int> val_indice = std::pair<double, int>(proj, indices);
                    pointsProject.push_back(val_indice);
                //}
            }
        }
    }
}

/**
 *  \brief Get Optimal position at which to place \p nbSamples 1D sample to minimize OT cost to the Radon transform of a \p D dimensional ball
 *
 *  A common result in optimal transport is that 1D placement comes from the inverse of the CDF of target distribution
 * @param nbSamples Number of samples to compute position for
 * @param pos Output buffer in which to put the positions
 */
template <class VECTYPE>
inline void getInverseRadonNCube(int D, int nbSamples, std::vector<double>& pos,VECTYPE dir, VECTYPE& center){

    pos.resize(nbSamples);
    std::vector<double> point_projected;
    for (int i = 0; i < nbSamples * mul; i++)
    {
        VECTYPE p;
        for (int d = 0; d < D; d++)
        {
            p[d] = ((float)rand() / RAND_MAX);
        }
        
        point_projected.push_back(toroidal_minus(p ,center)* dir);        
    }

    std::sort(point_projected.begin(),point_projected.end());
    for (int i = 0; i < nbSamples; i++)
    {
        pos[i] = point_projected[i*mul + int(mul/2)];
    }
}

/**
 * Choose \p m function of weighting.
 *
 * @param function_slices Table of function selection to output.
 * @param m Number of directions to pick
 */
inline void chooseFunctionSlices(std::vector<int>& function_slices, int m){
    int offset = rand()%(tileSize*tileSize);
    for (int k = 0; k < m; ++k){
        int a = (double(k)/(m/21.0)) * (tileSize*tileSize) + offset;
        function_slices[k] = a%(tileSize*tileSize);
    }
}

/**
 * Choose \p m function of weighting.
 *
 * @param function_slices Table of function selection to output.
 * @param m Number of directions to pick
 */
template <class VECTYPE>
inline void chooseWeightSlices(std::vector<double>& weight_slices, int m,VECTYPE c){
    for (int k = 0; k < m; ++k){
        c[0] = 10 + 0.5;
        c[1] = 10 + 0.5;
        double a = weight(c, c) * ((float)rand() / RAND_MAX);
        weight_slices[k] = a;
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
 * @param shift Output the 1D shift to apply to minimize transport cost
 */
template<class VECTYPE>
inline void slicedStepNCube(const VECTYPE& dir, int functionSelection,
                            double W_ref, const std::vector<VECTYPE>& points,
                            std::vector<double>& shift)
{
    int N = points.front().dim();
    VECTYPE center;
    for (int i = 0; i < N; ++i){
        center[i] = 0.0*((float)rand() / RAND_MAX);
    }
    for (size_t i = 0; i < points.size(); ++i){
        shift[i] = 0.0f;
    }
    std::vector<std::pair<double, int>> pointsProject;
    project(points, pointsProject, dir, center, functionSelection, W_ref, N);
    if(pointsProject.size() >= 1){
        //Compute optimal 1D position for given number of points;
        std::vector<double> pos(pointsProject.size());
        getInverseRadonNCube(N, pointsProject.size(), pos, dir, center);

        std::sort(pointsProject.begin(), pointsProject.end(), [](const std::pair<double, int> &x,
                                        const std::pair<double, int> &y)
        {
            return x.first < y.first;
        });
        double grad_normalization = float(pointsProject.size())/points.size();
        double normalise_factor = 1.0;
        //Computes required shift to optimize 1D optimal transport
        for (size_t i = 1; i < pointsProject.size()-1; ++i) {
            //Compute shifting
            double s = pos[i] - pointsProject[i].first;
            normalise_factor = ((pos[i + 1] - pos[i - 1]) / 2.0) * pointsProject.size();
            shift[pointsProject[i].second] += grad_normalization*(s/normalise_factor);
        }
        double s = pos[0] - pointsProject[0].first;
        normalise_factor = (pos[1] - pos[0]) * pointsProject.size();
        shift[pointsProject[0].second] += grad_normalization*(s / normalise_factor);

        s = pos[pointsProject.size() - 1] - pointsProject[pointsProject.size() - 1].first;
        normalise_factor = (pos[pointsProject.size() - 1] - pos[pointsProject.size() - 1 - 1]) * pointsProject.size();
        shift[pointsProject[pointsProject.size() - 1].second] += grad_normalization*(s / normalise_factor);
    }
}

/**
 * Compute optimal transport in 1D for the \p directions and displace \f$ x_j \f$ by
 * \f$\pmb{\delta}^j \EqDef \frac{1}{K} \sum_{i=1}^K d_{i,j}\, \theta_i \vspace*{-1mm}\f$ with
 * with \f$ d_{i,j} \f$ being the displacement of \f$\x^j\f$ that minimize the 1D sliced optimal
 * transport along direction \f$ \theta_i \f$
 *
 * @param pointsOut Table containing the points \f$ x_j \f$
 * @param directions Table containing the \f$\theta_i\f$
 * @param shift Used to avoid having to allocate uge chunks of memory. Must be a vector of size m containing vectors of same size as \p pointsOut.
 * @param finalShift Used to avoid having to allocate huge chunks of memory Must be a vector of same size as \p pointsOut.
 * @return the Wasserstein cost of the current iteration.
 */
template <class VECTYPE>
inline void slicedOptimalTransportBatchCube(std::vector<VECTYPE>& pointsOut,
                                 const std::vector<VECTYPE>& directions,
                                 const std::vector<int>& function_slices,
                                 const std::vector<double>& weight_slices,
                                 std::vector<std::vector<double>>& shift,
                                 std::vector<VECTYPE>& finalShift,
                                 std::vector<VECTYPE>& localShift)
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

        slicedStepNCube(dir,function_slices[k], weight_slices[k], pointsOut, shift[k]);
    }
        //Accumulate shift from all directions
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i) {
        VECTYPE sh(finalShift[i].dim());
        memset(&sh[0], 0, finalShift[i].dim() * sizeof(sh[0]));
        for (int k = 0; k < m; ++k) {
            sh += shift[k][i] * directions[k];
        }
        finalShift[i] = 1*(sh/m) + localShift[i]*0.0;
        localShift[i] = (sh/m) + localShift[i]*0.0;
    }

    //Displace points according to accumulated shift
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i) {
        pointsOut[i] += finalShift[i];

        for(int d = 0; d<finalShift[i].dim(); ++d){
            while(pointsOut[i][d]<0){
                pointsOut[i][d]+=1.0;
            }
            while(pointsOut[i][d]>1){
                pointsOut[i][d]-=1.0;
            }
        }
    }

}

void print_progress(double ratio){
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * ratio;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(ratio * 100.0) << " %\r";
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
inline void slicedOptimalTransportNCube(const std::vector<VECTYPE>& pointsIn,
                                        std::vector<VECTYPE>& pointsOut,
                                        int nbIter,
                                        int m,
                                        int seed,
                                        int tileS,
                                        bool silent)
{
    tileSize = tileS;
    int N = pointsIn.front().dim();
    pointsOut = pointsIn;

    //Accumulation shift to be applied later on
    std::vector<std::vector<double>> shift(m, std::vector<double>(pointsOut.size()));
    std::vector<VECTYPE> localshift(pointsOut.size(), VECTYPE(N));
    std::vector<VECTYPE> finalShift(pointsOut.size(), VECTYPE(N));

    std::vector<VECTYPE> directions(m, VECTYPE(N));
    std::vector<int> function_slices(m);
    std::vector<double> weight_slices(m);
    
    srand (seed);
    //Iterate 1D Optimal Transport
    for (int i = 0; i < nbIter; i += 1){
        if(!silent){
            print_progress(double(i)/nbIter);
        }
        VECTYPE c;
        chooseDirectionsND(directions, m, seed);
        chooseFunctionSlices(function_slices, m);
        chooseWeightSlices(weight_slices, m,c);

        slicedOptimalTransportBatchCube(pointsOut, directions, function_slices, weight_slices, shift, finalShift,localshift);
    }
    print_progress(1.0);

}


#endif //SLICEDOPTIM_NCUBESLICEDOPTIMALTRANSPORT_H
