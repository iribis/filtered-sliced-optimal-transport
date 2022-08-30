//
//

#ifndef IMAGE_BASED_SLICEDOPTIMALTRANSPORT_H
#define IMAGE_BASED_SLICEDOPTIMALTRANSPORT_H

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

const int multiplier_pointset_images = 3;
//const int w = 320;
//const int h = 400;

//const int w = 163;
//const int h = 215;

//const int w = 214;// groix chat
//const int h = 320;

//const int w = 600;// iliyan
//const int h = 600;

//const int w = 360;// land
//const int h = 640;

//const int w = 240;//elephants
//const int h = 180;

const int w = 188;//booby
const int h = 326;

float image[w][h][6];
float cumulativeImage[w][6];
float sum[6];

void readImage(){
    int row = 0, col = 0, numrows = 0, numcols = 0;
    for(int c=0; c<4;++c){
        //std::ifstream infile((std::string("../resources/groix_chat_")+std::to_string(c).c_str()+std::string(".pgm")).c_str());
        std::ifstream infile((std::string("../resources/land_")+std::to_string(c).c_str()+std::string(".pgm")).c_str());
        std::stringstream ss;
        std::string inputLine = "";

        // First line : version
        getline(infile,inputLine);
        if(inputLine.compare("P2") != 0) std::cerr << "Version error" << std::endl;
        else std::cout << "Version : " << inputLine << std::endl;

        // Second line : comment
        //getline(infile,inputLine);
        //std::cout << "Comment : " << inputLine << std::endl;

        // Continue with a stringstream
        ss << infile.rdbuf();
        // Third line : size
        ss >> numcols >> numrows;
        std::cout << numcols << " columns and " << numrows << " rows" << std::endl;

        // Following lines : data
        for(row = 0; row < numrows; ++row)
            for (col = 0; col < numcols; ++col) ss >> image[row][col][c];

        // Now print the array to see the result
        infile.close();

        for(row = 0; row < numrows; ++row){
            for (col = 0; col < numcols; ++col){
                //if(image[row][col][c]<10)
                //    image[row][col][c] = 0.0;
                //image[row][col][c] = 1.0 - image[row][col][c]/255.0;
            }

        }
        sum[c] = 0;
        for(row = 0; row < numrows; ++row){
            float cumulative = 0;
            for (col = 0; col < numcols; ++col){
                cumulative += image[row][col][c];
            }
            cumulativeImage[row][c] = cumulative;
            sum[c] += cumulative;
        }
    }

    //std::cout << (sum[0]/(sum[0]+sum[1]+sum[2]))<<std::endl;
    //std::cout << ((sum[0]+sum[1])/(sum[0]+sum[1]+sum[2]))<<std::endl;

    //double div = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]+sum[8];
    double div = sum[0]+sum[1]+sum[2]+sum[3];//+sum[4];//+sum[5];
    double tmp = 0;
    for (size_t i = 0; i < 4; i++)
    {
        tmp += sum[i];
        std::cout << tmp/div << std::endl;
    }
    
}



bool select(double indice_ratio, int selector){
    /*
    if(selector == 0){
        return indice_ratio < (sum[0]/(sum[0]+sum[1]+sum[2]));
    }else if(selector == 1){
        return indice_ratio < ((sum[0] + sum[1] )/(sum[0]+sum[1]+sum[2])) && indice_ratio > ((sum[0])/(sum[0]+sum[1]+sum[2]));
    }else if(selector == 4){
        return indice_ratio > ((sum[0] + sum[1] )/(sum[0]+sum[1]+sum[2]));
    }else if(selector == 2 || selector == 3){
        return indice_ratio < ((sum[0] + sum[1] )/(sum[0]+sum[1]+sum[2]));
    }else if(selector == 5 || selector == 6){
        return indice_ratio > ((sum[0])/(sum[0]+sum[1]+sum[2]));
    }else if(selector == 7 || selector == 8){
        return indice_ratio > ((sum[0] + sum[1] )/(sum[0]+sum[1]+sum[2])) || indice_ratio < (sum[0]/(sum[0]+sum[1]+sum[2]));
    }else{
        return true;
    }
    */
    //double div = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]+sum[8];
    double div = sum[0]+sum[1]+sum[2]+sum[3];//+sum[4];
    if(selector == 0){
        return indice_ratio < (sum[0]/(div));
    }else if(selector == 1){
        return indice_ratio < (sum[0] + sum[1] )/(div) && indice_ratio > ((sum[0])/(div));
    }else if(selector == 2){
        return indice_ratio < (sum[0] + sum[1] + sum[2])/(div) && indice_ratio > ((sum[0]+sum[1])/(div));
    }else if(selector == 3){
        return indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2])/(div));
    }else if(selector == 4){
        return indice_ratio < (sum[0] + sum[1])/(div);
    }else if(selector == 5){
        return indice_ratio < (sum[0])/(div) ||(indice_ratio < (sum[0] + sum[1] + sum[2])/(div) && indice_ratio > ((sum[0]+sum[1])/(div)));
    }else if(selector == 6){
        return indice_ratio < (sum[0])/(div) ||(indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2])/(div)));
    }else if(selector == 7){
        return (indice_ratio < (sum[0] + sum[1] )/(div) && indice_ratio > ((sum[0])/(div))) || (indice_ratio < (sum[0] + sum[1] + sum[2])/(div) && indice_ratio > ((sum[0]+sum[1])/(div)));
    }else if(selector == 8){
        return (indice_ratio < (sum[0] + sum[1] )/(div) && indice_ratio > ((sum[0])/(div)))||(indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2])/(div)));
    }else if(selector == 9){
        return (indice_ratio < (sum[0] + sum[1] + sum[2])/(div) && indice_ratio > ((sum[0]+sum[1])/(div)))||(indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2])/(div)));
    }else if(selector == 10){
        return indice_ratio < (sum[0] + sum[1] + sum[2])/(div);
    }else if(selector == 11){
        return (indice_ratio < (sum[0] + sum[1])/(div))||(indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2])/(div)));
    }else if(selector == 12){
        return indice_ratio > ((sum[0])/(div));
    }else if(selector == 13){
        return (indice_ratio < (sum[0])/(div))||(indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3])/(div) && indice_ratio > ((sum[0]+sum[1])/(div)));
    }
    /*else if(selector == 4){
        return indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3]+ sum[4])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2]+ sum[3])/(div));
    }else if(selector == 5){
        return indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3]+ sum[4]+ sum[5])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2]+ sum[3]+ sum[4])/(div));
    }else if(selector == 6){
        return indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3]+ sum[4]+ sum[5]+ sum[6])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2]+ sum[3]+ sum[4]+ sum[5])/(div));
    }else if(selector == 7){
        return indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3]+ sum[4]+ sum[5]+ sum[6]+ sum[7])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2]+ sum[3]+ sum[4]+ sum[5]+ sum[6])/(div));
    }else if(selector == 8){
        return indice_ratio < (sum[0] + sum[1] + sum[2] + sum[3]+ sum[4]+ sum[5]+ sum[6]+ sum[7]+ sum[8])/(div) && indice_ratio > ((sum[0]+sum[1]+ sum[2]+ sum[3]+ sum[4]+ sum[5]+ sum[6]+ sum[7])/(div));
    }*/else{
        return true;
    }
}

int select_image(int selector){
    if(selector<4){
        return selector;
    }else if(selector == 4){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[1];
        if(rnd<sum[0]/div){
            return 0;
        }else{
            return 1;
        }
    }else if(selector == 5){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[2];
        if(rnd<sum[0]/div){
            return 0;
        }else{
            return 2;
        }
    }else if(selector == 6){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[3];
        if(rnd<sum[0]/div){
            return 0;
        }else{
            return 3;
        }
    }else if(selector == 7){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[1]+sum[2];
        if(rnd<sum[1]/div){
            return 1;
        }else{
            return 2;
        }
    }else if(selector == 8){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[1]+sum[3];
        if(rnd<sum[1]/div){
            return 1;
        }else{
            return 3;
        }
    }else if(selector == 9){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[2]+sum[3];
        if(rnd<sum[2]/div){
            return 2;
        }else{
            return 3;
        }
    }else if(selector == 10){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[1]+sum[2];
        if(rnd<sum[0]/div){
            return 0;
        }else if(rnd<(sum[0]+sum[1])/div){
            return 1;
        }else{
            return 2;
        }
    }else if(selector == 11){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[1]+sum[3];
        if(rnd<sum[0]/div){
            return 0;
        }else if(rnd<(sum[0]+sum[1])/div){
            return 1;
        }else{
            return 3;
        }
    }else if(selector == 12){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[1]+sum[2]+sum[3];
        if(rnd<sum[1]/div){
            return 1;
        }else if(rnd<(sum[1]+sum[2])/div){
            return 2;
        }else{
            return 3;
        }
    }else if(selector == 13){
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[2]+sum[3];
        if(rnd<sum[0]/div){
            return 0;
        }else if(rnd<(sum[0]+sum[2])/div){
            return 2;
        }else{
            return 3;
        }
    }else{
        float rnd = ((float)rand() / RAND_MAX);
        float div = sum[0]+sum[1]+sum[2]+sum[3];
        if(rnd<sum[0]/div){
            return 0;
        }else if(rnd<(sum[0]+sum[1])/div){
            return 1;
        }else if(rnd<(sum[0]+sum[1]+sum[2])/div){
            return 2;
        }else{
            return 3;
        }
    }
    /*
    if(selector == 0){
        return 0;
    }else if(selector == 1){
        return 1;
    }else if(selector == 4){
        return 2;
    }else if(selector == 2 || selector == 3){
        return rand()%2;
    }else if(selector == 5 || selector == 6){
        return 1+rand()%2;
    }else if(selector == 7 || selector == 8){
        int rnd = rand()%2;
        if(rnd == 1){
            return 2;
        }else{
            return 0;
        }
    }else{
        return rand()%3;
    }*/
}

template <class VECTYPE>
void project(const std::vector<VECTYPE>& points, std::vector<std::pair<double, int>> &pointsProject, const VECTYPE& dir,int N, VECTYPE& offset, int selector){
    for(size_t s = 0; s < points.size(); ++s){
        if(select(s/float(points.size()),selector)){
            VECTYPE p = points[s];
            //double proj = (toroidal_minus(p, offset) * dir);
            double proj = (p * dir);
            std::pair<double, int> val_indice = std::pair<double, int>(proj, s);
            pointsProject.push_back(val_indice);
        }
    }

}

template <class VECTYPE>
inline void getInverseImage(int D, int nbSamples, std::vector<double>& pos,VECTYPE dir,VECTYPE offset, int selector){

    pos.resize(nbSamples);
    std::vector<double> posbis;
    posbis.resize(nbSamples*multiplier_pointset_images);
    for (int i = 0; i < nbSamples*multiplier_pointset_images; ++i){
        int img = select_image(selector);
        VECTYPE p;
        float rnd1 = ((float)rand() / RAND_MAX)*sum[img];
        float cum = 0;
        float x = 0;
        float y = 0;
        for (int j = 0; j < w; j++)
        {
            if(cum+cumulativeImage[j][img]>=rnd1){
                x = j;
                break;
            }
            cum+=cumulativeImage[j][img];
        }
        
        float rnd2 = ((float)rand() / RAND_MAX)*cumulativeImage[int(x)][img];
        cum = 0;
        for (int j = 0; j < h; j++)
        {
            if(cum+image[int(x)][j][img]>=rnd2){
                y = j;
                break;
            }
            cum+=image[int(x)][j][img];
        }

        p[1] = (1-(x+((float)rand() / RAND_MAX))/w)*w/std::max(w,h);
        p[0] = ((y+((float)rand() / RAND_MAX))/h)*h/std::max(w,h);

        posbis[i] = p*dir;
    }
    std::sort(posbis.begin(),posbis.end());
    for (int i = 0; i < nbSamples; i++)
    {
        pos[i] = posbis[i*multiplier_pointset_images + int(multiplier_pointset_images/2)];
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
inline void slicedStepNImageBased(const VECTYPE& dir,
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
    project(points, pointsProject, dir, N,offset,selector);

    std::vector<double> pos(pointsProject.size());
    getInverseImage(N,pointsProject.size(), pos,dir,offset,selector);

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
inline void slicedOptimalTransportBatch_image_based(std::vector<VECTYPE>& pointsOut,
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

        slicedStepNImageBased(dir, pointsOut, shift[k]);
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
        //for(int d = 0; d<finalShift[i].dim(); ++d){
        //    while(pointsOut[i][d]<0){
        //        pointsOut[i][d]+=1.0;
        //    }
        //    while(pointsOut[i][d]>1){
        //        pointsOut[i][d]-=1.0;
        //    }
        //}
    }

}

void print_progress_image_based(double ratio){
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
inline void slicedOptimalTransportNImageBased(const std::vector<VECTYPE>& pointsIn,
                                        std::vector<VECTYPE>& pointsOut,
                                        int nbIter,
                                        int m,
                                        int seed, bool silent = false)
{

    int N = pointsIn.front().dim();
    pointsOut = pointsIn;
    readImage();
    //Accumulation shift to be applied later on
    std::vector<std::vector<double>> shift(m, std::vector<double>(pointsOut.size()));
    std::vector<VECTYPE> finalShift(pointsOut.size(), VECTYPE(N));

    std::vector<VECTYPE> directions(m, VECTYPE(N));

    //Iterate 1D Optimal Transport
    for (int i = 0; i < nbIter; i += 1){
        if(!silent){
            print_progress_image_based(double(i)/nbIter);
        }
        chooseDirectionsND(directions, m, seed);

        slicedOptimalTransportBatch_image_based(pointsOut, directions, shift, finalShift);
    }
    print_progress(1.0);

}

#endif
