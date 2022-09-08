//
//

#ifndef SLICEDOPTIM_MY_UTILITY_H
#define SLICEDOPTIM_MY_UTILITY_H

#include <random>
#include <functional>
#include <string>
#include "../Math/VecX.h"

const double PI =3.141592653589793238463;

template<class VECTYPE, class RandomGenerator>
inline VECTYPE randomVectorInBall(int dim, RandomGenerator &engine) {
    std::normal_distribution<double> normalDistribution(0, 1);
    std::uniform_real_distribution<double> unif(0, 1);
    VECTYPE v(dim);

    for (int j = 0; j < dim; ++j) {
        v[j] = normalDistribution(engine);
    }
    v.normalize();
    v *= std::pow(unif(engine), 1. / dim);

    return v;
}

template<class VECTYPE, class RandomGenerator>
inline VECTYPE randomVectorInCube(int dim, RandomGenerator &engine) {
    std::uniform_real_distribution<double> unif(0, 1);
    VECTYPE v(dim);

    for (int j = 0; j < dim; ++j) {
        v[j] = unif(engine);
    }

    return v;
}

double clamp(double v, double min, double max);

double inverseFunction(std::function<double(double)> &f, std::function<double(double)> &df, double v);

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <class VECTYPE>
VECTYPE toroidal_minus(VECTYPE& v1, VECTYPE& v2){
    VECTYPE res(v1.dim());
    for(int i=0; i<v1.dim();++i){
        res[i] = v1[i] - v2[i];
        if(1.0 > (v1[i] - v2[i] + 1) && 0.0 <= (v1[i] - v2[i] + 1)){
            res[i] = v1[i] - v2[i] + 1;
        }
        if(1.0 > (v1[i] - v2[i] - 1) && 0.0 <= (v1[i] - v2[i] - 1)){
            res[i] = v1[i] - v2[i] - 1;
        }
    }
    return res;
}

template <class VECTYPE>
float toroidal_norm(VECTYPE& v1, VECTYPE& v2){
    VECTYPE res(v1.dim());
    for(int i=0; i<v1.dim();++i){
        res[i] = v1[i] - v2[i];
        if(std::pow(res[i],2) > std::pow(v1[i] - v2[i] + 1,2)){
            res[i] = v1[i] - v2[i] + 1;
        }
        if(std::pow(res[i],2) > std::pow(v1[i] - v2[i] - 1,2)){
            res[i] = v1[i] - v2[i] - 1;
        }
    }
    return res.norm();
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
inline void chooseDirectionsND(std::vector<VECTYPE>& directions, int m, int seed){

    static std::mt19937 generatorND(seed);
    static std::normal_distribution<>normalND;
    static std::uniform_real_distribution<double> unif(0, 1);

    int dim = directions.front().dim();

    for (int k = 0; k < m; ++k){
        if(dim == 2){
            
            double rnd =  unif(generatorND);
            if(rnd < 1.3){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][0] = cos(theta);
                directions[k][1] = sin(theta);
            }else if(rnd < 0.65){
                directions[k][0] = 1.0;
                directions[k][1] = 0.0;
            }else{
                directions[k][0] = 0.0;
                directions[k][1] = 1.0;
            }
        }else if(dim == 3){
            double rnd =  unif(generatorND);
            if(rnd < 0.7){
                for (int j = 0; j < dim; ++j){
                    directions[k][j] = normalND(generatorND);
                }
            }else if(rnd < 1.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][0] = cos(theta);
                directions[k][1] = sin(theta);
                directions[k][2] = 0.0;
            }else if(rnd < 0.8){
                directions[k][0] = 1.0;
                directions[k][1] = 0.0;
                directions[k][2] = 0.0;
            }else if(rnd < 0.9){
                directions[k][0] = 0.0;
                directions[k][1] = 1.0;
                directions[k][2] = 0.0;
            }else{
                directions[k][0] = 0.0;
                directions[k][1] = 0.0;
                directions[k][2] = 1.0;
            }
        }else if(dim == 4){
            double rnd =  unif(generatorND)*1+1.0;
            directions[k][0] = 0.0;
            directions[k][1] = 0.0;
            directions[k][2] = 0.0;
            directions[k][3] = 0.0;
            if(rnd < 0.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][0] = cos(theta);
                directions[k][1] = sin(theta);
            }else if(rnd < 0.85){
                directions[k][0] = 1.0;
                directions[k][1] = 0.0;
            }else if(rnd < 1.0){
                directions[k][0] = 0.0;
                directions[k][1] = 1.0;
            }
            if(rnd < 1.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][2] = cos(theta);
                directions[k][3] = sin(theta);
            }else if(rnd < 1.85){
                directions[k][2] = 1.0;
                directions[k][3] = 0.0;
            }else {
                directions[k][2] = 0.0;
                directions[k][3] = 1.0;
            }
            //std::cout <<  directions[k] << std::endl;
        }else if(dim == 8){
            
            double rnd =  unif(generatorND)*4;
            directions[k][0] = 0.0;
            directions[k][1] = 0.0;
            directions[k][2] = 0.0;
            directions[k][3] = 0.0;
            directions[k][4] = 0.0;
            directions[k][5] = 0.0;
            directions[k][6] = 0.0;
            directions[k][7] = 0.0;
            if(rnd < 0.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][0] = cos(theta);
                directions[k][1] = sin(theta);
            }else if(rnd < 0.85){
                directions[k][0] = 1.0;
                directions[k][1] = 0.0;
            }else if(rnd < 1.0){
                directions[k][0] = 0.0;
                directions[k][1] = 1.0;
            }
            if(rnd < 1.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][2] = cos(theta);
                directions[k][3] = sin(theta);
            }else if(rnd < 1.85){
                directions[k][2] = 1.0;
                directions[k][3] = 0.0;
            }else if(rnd < 2.0){
                directions[k][2] = 0.0;
                directions[k][3] = 1.0;
            }
            if(rnd < 2.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][4] = cos(theta);
                directions[k][5] = sin(theta);
            }else if(rnd < 2.85){
                directions[k][4] = 1.0;
                directions[k][5] = 0.0;
            }else if(rnd < 3.0){
                directions[k][4] = 0.0;
                directions[k][5] = 1.0;
            }
            if(rnd < 3.7){
                double theta =  (float(k)/m + unif(generatorND)/float(m))*2*PI; // stratified 2D directions
                directions[k][6] = cos(theta);
                directions[k][7] = sin(theta);
            }else if(rnd < 3.85){
                directions[k][6] = 1.0;
                directions[k][7] = 0.0;
            }else {
                directions[k][6] = 0.0;
                directions[k][7] = 1.0;
            }
        }else{
            for (int j = 0; j < dim; ++j){
                directions[k][j] = normalND(generatorND);
            }
        }   
        directions[k].normalize();
    }
}

template <class VECTYPE>
void export_sampler(const std::vector<VECTYPE>& points, std::string filename, int tile_size, int spp){
    int dim = points.front().dim();
    std::ofstream myfile (filename,std::ios::trunc);
    if (myfile.is_open())
    {
        myfile << "#pragma once\n\n\n";
        myfile << "const float tile["<<tile_size<<"]["<<tile_size*spp*dim<<"] = {";
        for (size_t i = 0; i < tile_size; i++)
        {
            if(i==0){myfile <<"{";}
            else{myfile <<",{";}
            
            for (size_t j = 0; j < tile_size; j++)
            {
                
                for (size_t k = 0; k < spp; k++)
                {
                    for (size_t d = 0; d < dim; d++)
                    {
                        if(j==0 && k == 0 && d==0){myfile <<points[(i*tile_size+j)*spp+k][d];}
                        else{myfile <<","<<points[(i*tile_size+j)*spp+k][d];}
                    }
                }
                
            }
            myfile <<"}";
        }
        myfile <<"};";
        myfile <<"\n\n\n";
        myfile <<"float sample(int i, int j, int s, int d){\n";
        myfile <<"\treturn tile[i%"<<tile_size<<"][(j%"<<tile_size<<")*"<<spp<<"*"<<dim<<"+(s%"<<spp<<")*"<<dim<<"+(d%"<<dim<<")];\n";
        myfile <<"}\n";
        myfile.close();
    }
    
}



#endif //SLICEDOPTIM_MY_UTILITY_H
