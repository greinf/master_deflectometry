#include "GoldsteinWrapper.hpp"

extern "C" {
#include "grad.h"
#include "util.h"
#include "pi.h"
#include "extract.h"
#include <omp.h>
}
#define BORDER 0x20   /* 6th bit */
// Full C→C++ wrapper with optional mask
void goldsteinUnwrapCV(const cv::Mat& wrapped, cv::Mat& unwrapped, const cv::Mat& mask)
{
    CV_Assert(wrapped.type() == CV_32F);
    CV_Assert(mask.empty() || (mask.size() == wrapped.size()));

    const int W = wrapped.cols, H = wrapped.rows;
    const int N = W * H;

    float* phase01, * soln01, * gradx, * grady;
    unsigned char* bitflags;
    int* path_order, * list;

    AllocateFloat(&phase01, N, "phase01");
    AllocateFloat(&soln01, N, "soln01");
    AllocateFloat(&gradx, N, "gradx");
    AllocateFloat(&grady, N, "grady");
    AllocateByte(&bitflags, N, "bitflags");
    AllocateInt(&path_order, N, "path_order");
    AllocateInt(&list, 2 * (W + H), "list");

    // Convert cv::Mat (radians) -> normalized [0,1)
    const float inv_twopi = 1.0f / (2.0f * static_cast<float>(CV_PI));
    const float* src = wrapped.ptr<float>();

    const bool hasMask = !mask.empty();
    const unsigned char* mask8 = (mask.type() == CV_8U) ? mask.ptr<uchar>() : nullptr;
    const float* mask32 = (mask.type() == CV_32F) ? mask.ptr<float>() : nullptr;

    for (int k = 0; k < N; ++k) {
        float r = src[k];
        if (r < 0.0f)
            r += 2.0f * static_cast<float>(CV_PI);
        phase01[k] = r * inv_twopi;

        // Set border flag for invalid pixels
        if (hasMask) {
            bool valid = (mask8 ? (mask8[k] > 0) : (mask32[k] > 0.5f));
            bitflags[k] = valid ? 0 : BORDER;
        }
        else {
            bitflags[k] = 0;
        }
    }

    // Set OpenMP threads (optional)
    NUM_CORES = 1;
    omp_set_num_threads(NUM_CORES);

    // Run the Goldstein pipeline
    Gradxy(phase01, gradx, grady, W, H);
    int numRes = Residues_parallel(phase01, bitflags, W, H);
    int maxCutLen = (W + H) / 2;
    GoldsteinBranchCuts_parallel(bitflags, maxCutLen, numRes, W, H);

    int numPieces = UnwrapAroundCutsFrontier(
        phase01, bitflags, soln01, W, H, path_order, grady, gradx, list, N);

    std::cout << "[Goldstein] residues=" << numRes
        << ", connected regions=" << numPieces << std::endl;

    // Copy back and scale to radians
    unwrapped.create(H, W, CV_32F);
    float* dst = unwrapped.ptr<float>();
    const float twopi = 2.0f * static_cast<float>(CV_PI);
    for (int k = 0; k < N; ++k)
        dst[k] = soln01[k] * twopi;

    // Free memory
    free(phase01);
    free(soln01);
    free(gradx);
    free(grady);
    free(bitflags);
    free(path_order);
    free(list);
}
