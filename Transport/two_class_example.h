//
//

#ifndef two_class_SAMPLER_OPTIMALTRANSPORT_H
#define two_class_SAMPLER_OPTIMALTRANSPORT_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstring>
#include "../Math/VecX.h"
#include "../Math/myMath.h"
#include "../Tools/iopointset.h"
#include "../Tools/my_utility.h"

double multiplier_two_class = 3;

template <class VECTYPE>
VECTYPE toroidal_minus_two_class(VECTYPE& v1, VECTYPE& v2) {
    VECTYPE res(v1.dim());
    for (int i = 0; i < v1.dim(); ++i) {
        res[i] = v1[i] - v2[i];
        if (1.0 > (v1[i] - v2[i] + 1) && 0.0 <= (v1[i] - v2[i] + 1)) {
            res[i] = v1[i] - v2[i] + 1;
        } else if (1.0 > (v1[i] - v2[i] - 1) && 0.0 <= (v1[i] - v2[i] - 1)) {
            res[i] = v1[i] - v2[i] - 1;
        }
    }
    return res;
}

template <class VECTYPE>
float toroidal_norm_two_class(VECTYPE& v1, VECTYPE& v2){
    VECTYPE res(v1.dim());
    for(int i=0; i<v1.dim();++i){
        res[i] = v1[i] - v2[i];
        if (1.0 > (v1[i] - v2[i] + 1) && 0.0 <= (v1[i] - v2[i] + 1)) {
            res[i] = v1[i] - v2[i] + 1;
        } else if (1.0 > (v1[i] - v2[i] - 1) && 0.0 <= (v1[i] - v2[i] - 1)) {
            res[i] = v1[i] - v2[i] - 1;
        }
    }
    return res.norm();
}


template <class VECTYPE>
void project_two_class(const std::vector<VECTYPE>& points,
            std::vector<std::pair<double, int>>& pointsProject,
            const VECTYPE& dir, VECTYPE& center, int min_index, int max_index, int dim){



    for(int s = min_index; s < max_index; s++){
 
        VECTYPE p = points[s];
        double proj = (toroidal_minus_two_class(p,center) * dir);
        int indices = s;
        std::pair<double, int> val_indice = std::pair<double, int>(proj, indices);
        pointsProject.push_back(val_indice);
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
inline void getInverseRadonNCube_two_class(int D, int nbSamples, std::vector<double>& pos,VECTYPE dir, VECTYPE& center){

    pos.resize(nbSamples);
    std::vector<double> point_projected;
    //point_projected.resize(nbSamples * multiplier_two_class);
    for (int i = 0; i < nbSamples * multiplier_two_class; i++)
    {
        VECTYPE p;
        for (int d = 0; d < D; d++)
        {
            p[d] = ((float)rand() / RAND_MAX);
        }
        
        point_projected.push_back(toroidal_minus_two_class(p,center) * dir);        
    }

    std::sort(point_projected.begin(),point_projected.end());
    for (int i = 0; i < nbSamples; i++)
    {
        pos[i] = point_projected[i*multiplier_two_class + int(multiplier_two_class/2)];
    }
}

/**
 * Choose \p m directions in N dimension N being defined by the dimention of the content of directions.
 * Two selection methods are available. Either the direction are uniformly selected in ND
 * or one can force half of the them to lie in 2D planes to optimize the projection repartition as well.
 *
 * @param directions Table of directions to output. Must be initialized with \p m VECTYPE of the disired dimension
 * @param m Number of directions to pick
 * @param seed Seed for the random generator. Only applied once
 * @param projective If true half the directions will lie in 2D planes.
 */
template <class VECTYPE>
inline void chooseDirectionsND_two_class(std::vector<VECTYPE>& directions, int m, int seed){

    static std::mt19937 generatorND(seed);
    static std::normal_distribution<>normalND;
    static std::uniform_real_distribution<double> unif(0, 1);

    int dim = directions.front().dim();

    for (int k = 0; k < m; ++k){
        if(dim == 2){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][0] = cos(theta);
                directions[k][1] = sin(theta);
        }else{
            for (int j = 0; j < dim; ++j){
                directions[k][j] = normalND(generatorND);
            }
            
        }   
        directions[k].normalize();
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
inline void slicedStepNCube_two_class(const VECTYPE& dir, const std::vector<VECTYPE>& points,
                            std::vector<double>& shift)
{
    int N = points.front().dim();
    VECTYPE center;
    for (int i = 0; i < N; ++i){
        center[i] = 1.0*((float)rand() / RAND_MAX);
    }
    for (size_t i = 0; i < points.size(); ++i){
        shift[i] = 0.0f;
    }

    float tmp =((float)rand() / RAND_MAX);
    int min = 0;
    int max = 0;
    if(tmp<0.25){
        min = 0;
        max = points.size()/2;
    }else if(tmp<0.5){
        min = points.size()/2;
        max = points.size();
    }else{
        min = 0;
        max = points.size();
    }
    
    std::vector<std::pair<double, int>> pointsProject;
    project_two_class(points, pointsProject, dir, center, min, max, N);


    //Compute optimal 1D position for given number of points;
    std::vector<double> pos(pointsProject.size());
    getInverseRadonNCube_two_class(N, pointsProject.size(), pos, dir, center);

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
        normalise_factor = ((pos[i + 1] - pos[i - 1]) / 2.0) * (pointsProject.size());
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
 * @param shift Used to avoid having to allocate uge chunks of memory. Must be a vector of size m containing vectors of same size as \p pointsOut.
 * @param finalShift Used to avoid having to allocate huge chunks of memory Must be a vector of same size as \p pointsOut.
 * @return the Wasserstein cost of the current iteration.
 */
template <class VECTYPE>
inline void slicedOptimalTransportBatchCube_two_class(std::vector<VECTYPE>& pointsOut,
                                 const std::vector<VECTYPE>& directions,
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

        slicedStepNCube_two_class(dir, pointsOut, shift[k]);
    }
        //Accumultiplierate shift from all directions
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i) {
        VECTYPE sh(finalShift[i].dim());
        memset(&sh[0], 0, finalShift[i].dim() * sizeof(sh[0]));
        for (int k = 0; k < m; ++k) {
            sh += shift[k][i] * directions[k];
        }
        finalShift[i] = (sh/m) + localShift[i]*0.0;
        localShift[i] = (sh/m) + localShift[i]*0.0;
    }

    //Displace points according to accumultiplierated shift
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

void print_progress_two_class(double ratio){
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
inline void slicedOptimalTransportNCube_two_class(const std::vector<VECTYPE>& pointsIn,
                                        std::vector<VECTYPE>& pointsOut,
                                        int nbIter,
                                        int m,
                                        int seed,
                                        int tileS,
                                        bool silent, 
                                        int subdiv)
{
    tileSize = tileS;
    int N = pointsIn.front().dim();
    pointsOut = pointsIn;

    //Accumultiplieration shift to be applied later on
    std::vector<std::vector<double>> shift(m, std::vector<double>(pointsOut.size()));
    std::vector<VECTYPE> localshift(pointsOut.size(), VECTYPE(N));
    std::vector<VECTYPE> finalShift(pointsOut.size(), VECTYPE(N));

    std::vector<VECTYPE> directions(m, VECTYPE(N));

    srand (seed);
    //Iterate 1D Optimal Transport
    for (int i = 0; i < nbIter; i += 1){
        if(!silent){
            print_progress_two_class(double(i)/nbIter);
        }
        chooseDirectionsND_two_class(directions, m, seed);

        slicedOptimalTransportBatchCube_two_class(pointsOut, directions, shift, finalShift,localshift);
    }
    print_progress(1.0);

}


#endif //SLICEDOPTIM_NCUBESLICEDOPTIMALTRANSPORT_H
