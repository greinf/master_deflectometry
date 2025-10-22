#pragma once
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

    /* from grad.c */
    void Gradxy(float* phase, float* gradx, float* grady, int xsize, int ysize);

    /* from main (renamed to goldstein.c) */
    int Residues_parallel(float* phase, unsigned char* bitflags, int xsize, int ysize);
    void GoldsteinBranchCuts_parallel(unsigned char* bitflags, int MaxCutLen, int NumRes, int xsize, int ysize);
    int UnwrapAroundCutsFrontier(float* phase,
        unsigned char* bitflags,
        float* soln,
        int xsize,
        int ysize,
        int* path_order,
        float* grady,
        float* gradx,
        int* list,
        int length);

    /* global used by GoldsteinBranchCuts_parallel */
    extern int NUM_CORES;

#ifdef __cplusplus
}
#endif

void goldsteinUnwrapCV(const cv::Mat& wrapped, cv::Mat& unwrapped, const cv::Mat& mask = cv::Mat());