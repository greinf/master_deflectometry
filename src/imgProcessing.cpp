#include "imgProcessing.hpp"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <math.h>
#include <imageHandler.hpp>
#include <opencv2/phase_unwrapping/histogramphaseunwrapping.hpp>
#include "GoldsteinWrapper.hpp"
#include <filesystem>
#include <array>
#include <vector>
#include <cctype>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/traits.hpp>


//Allocator!!!! must be fixed 
//
//template <typename T, std::size_t Alignment = 32>
//struct AlignedAllocator {
//    using value_type = T;
//
//    AlignedAllocator() noexcept = default;
//    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
//
//    T* allocate(std::size_t n) {
//        void* ptr = nullptr;
//        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
//            throw std::bad_alloc();
//        return reinterpret_cast<T*>(ptr);
//    }
//
//    void deallocate(T* p, std::size_t) noexcept {
//        free(p);
//    }
//};

auto normalizeAndDisplay = [](const cv::Mat& img) -> void {
    cv::Mat norm;
    cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Normalized", norm);
    cv::waitKey(0);
    //cv::destroyWindow("Normalized");
    };


//Debugging function to calculate gradient strength
auto grad_strength = [](const cv::Mat& f) -> std::pair<cv::Mat, cv::Mat> {
    cv::Mat fx, fy;
    std::pair<cv::Mat, cv::Mat> pair;
    cv::Sobel(f, fx, CV_32F, 1, 0, 3); // d/dx
    cv::Sobel(f, fy, CV_32F, 0, 1, 3); // d/dy
    pair.first = fx;
    pair.second = fy;
    
    return pair;
    };

ImageProcessing::minmaxloc ImageProcessing::get_minmaxloc(cv::Mat& mat) const {
    minmaxloc helper;
    cv::minMaxLoc(mat, &helper.minval, &helper.maxval, &helper.minloc, &helper.maxloc);
    return helper;
}




std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> ImageProcessing::do_calibration_Points(
    const std::vector<cv::Mat>& unwrapped,
    const cv::Mat& mask,
    const double wavelength,
    const int gridX,
    const int gridY,
    const double pixe_pitch_mm,
    const double screenWidth_mm,
    const double screenHeight_mm) 
{
    CV_Assert(unwrapped.size() == 2);
    CV_Assert(unwrapped[0].type() == CV_64F);
    CV_Assert(unwrapped[0].size() == unwrapped[1].size());
    CV_Assert(unwrapped[0].size() == mask.size());
    CV_Assert(gridX >= 0 && gridY >= 0);
    CV_Assert(screenWidth_mm > 0 && screenHeight_mm > 0);

    std::vector<cv::Vec2d> imagePoints;
    std::vector<cv::Vec3d> objectPoints;
    std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> output;


    cv::Size sz = unwrapped[0].size();
    const double step_X = static_cast<double>(sz.width) / gridX;
    const double step_Y = static_cast<double>(sz.height) / gridY;
    //Debug image
    cv::Mat debug(mask.clone());

    cv::cvtColor(debug, debug, cv::COLOR_GRAY2BGR);

    for (double rowd = 0; rowd < sz.height; rowd += step_Y) {
        int row = std::min(int(rowd), sz.height - 1);
        const double* x_ptr = unwrapped[0].ptr<double>(row);
        const double* y_ptr = unwrapped[1].ptr<double>(row);
        const uchar* mask_ptr = mask.ptr<uchar>(row);
        for (double colsd = 0; colsd < sz.width; colsd += step_X) {
            int cols = std::min(int(colsd), sz.width - 1);
            if (mask_ptr[cols] == 0) continue;
            if (row < 0 || cols < 0) throw std::runtime_error("Index out of bounds: index < 0");
            if (row >= sz.height || cols >= sz.width) throw std::runtime_error("Index out of bounds : index > 0");

            //double theoretical_limitX{ static_cast<double>(wavelength * sz.width) };
            //double theoretical_limitY{ static_cast<double>(wavelength * sz.height) };

            double phaseValX = x_ptr[cols];
            double phaseValY = y_ptr[cols];

            double X_pixel = (phaseValX / CV_2PI) * wavelength;
            double Y_pixel = (phaseValY / CV_2PI) * wavelength;

            double X_mm = X_pixel * pixe_pitch_mm;
            double Y_mm = Y_pixel * pixe_pitch_mm;
            imagePoints.emplace_back(cv::Vec2d(cols, row));
            objectPoints.emplace_back(cv::Vec3d(X_mm, Y_mm, 0));
            //std::cout << "ImagePoints: " << cv::Vec2d(cols, row) << '\n';
            //std::cout << "ObjectPoints: " << cv::Vec3d(X_mm, Y_mm, 0) << '\n';
            cv::drawMarker(debug, cv::Point(cols, row), cv::Scalar(0, 255, 0), cv::MARKER_CROSS);
        }
    }
    output.first = imagePoints;
    output.second = objectPoints;

    cv::normalize(debug, debug, 0, 255, cv::NORM_MINMAX, CV_8SC3);
    cv::imshow("debug", debug);
    cv::waitKey(0);
    return output;
}


std::vector<double> ImageProcessing::extract_Column(const cv::Mat& picture, const cv::Mat& mask, int col) {
    CV_Assert(picture.size() == mask.size());
    CV_Assert(picture.type() == CV_64FC1);
    CV_Assert(picture.channels() == 1);
    CV_Assert(mask.type() == CV_8U);

    if(!col) col = (picture.cols / 2);
           
    std::vector<double> column_vec;

    for (int row = 0; row < picture.rows; ++row) {
        const double* ptr_pic = picture.ptr<double>(row);
        const uchar* ptr_mask = mask.ptr<uchar>(row);
        if (*ptr_mask) column_vec.push_back(ptr_pic[col]);
    }

    return column_vec;
}


std::vector<double> ImageProcessing::extract_Line(const cv::Mat& picture, const cv::Mat& mask, int row) {
    CV_Assert(picture.size() == mask.size());
    CV_Assert(picture.type() == CV_64FC1);
    CV_Assert(picture.channels() == 1);
    CV_Assert(mask.type() == CV_8U);

    if (!row) row = (picture.rows / 2);

    std::vector<double> column_vec;

    for (int row = 0; row < picture.rows; ++row) {
        const double* ptr_pic = picture.ptr<double>(row);
        const uchar* ptr_mask = mask.ptr<uchar>(row);
        if (*ptr_mask) column_vec.push_back(ptr_pic[row]);
    }

    return column_vec;
}

std::pair<double, double> ImageProcessing::fitLine1D(const std::vector<double>& y) {
    CV_Assert(!y.empty());
    const std::size_t N = static_cast<std::size_t>(y.size());
    double sumx = 0.0, sumy = 0.0, sumxx = 0.0, sumxy = 0.0;
    
    for (std::size_t i = 0; i < N; ++i) {
    	double x = static_cast<double>(i);
    	double v = y[i];
    	sumx += x;
    	sumy += v;
    	sumxx += x * x;
    	sumxy += x * v;
    }
    
    double denom = N * sumxx - sumx * sumx;
    double a = (N * sumxy - sumx * sumy) / denom;
    double b = (sumy - a * sumx) / N;
    
    return { a, b };
    }

cv::Mat ImageProcessing::do_reprojection_error(
    const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>>& caliPoints,
    const cv::Mat& caliMatrix,
    const cv::Mat& distCoeffs,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    double* sqrtErr,
    double* maxErr,
    const cv::Mat& mask)
{
    CV_Assert(caliPoints.first.size() == caliPoints.second.size());
    CV_Assert(caliMatrix.type() == CV_64F);
    CV_Assert(rvec.type() == CV_64F);
    CV_Assert(tvec.type() == CV_64F);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(sqrtErr && maxErr);

    // --- Extrinsics ---
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat R_inv = R.inv();

    // --- Camera Matrix ---
    cv::Mat Cam_inv{ caliMatrix.inv() };

    // Camera center in world/display coords: C = -R^-1 t
    cv::Mat C_mat = -R_inv * tvec;
    cv::Vec3d C(
        C_mat.at<double>(0),
        C_mat.at<double>(1),
        C_mat.at<double>(2));

    // --- Prepare undistorted normalized image points ---
    std::vector<cv::Point2d> img_pts;
    img_pts.reserve(caliPoints.first.size());
    for (const auto& v : caliPoints.first)
        img_pts.emplace_back(v[0], v[1]);

    std::vector<cv::Point2d> undist;

    cv::Size dist_sz = distCoeffs.size();
    const double* dist_ptr = distCoeffs.ptr<double>(0);
    // ---Check if distotion_coefficients are set. Mainly debugging ---
    if (dist_ptr[0] || dist_ptr[1] || dist_ptr[2] || dist_ptr[3]) {
        cv::undistortImagePoints(img_pts, undist, caliMatrix, distCoeffs);
    }

    else undist = img_pts;

    // --- Error map (dx, dy) on display plane ---
    cv::Mat error_map(mask.size(), CV_64FC2, cv::Scalar(0, 0));

    double sumsq = 0.0;
    *maxErr = 0.0;
    std::size_t N = 0;

    // Assume display plane at Z = 0 in world coordinates:
    const double Z0 = 0.0;

    for (std::size_t i = 0; i < caliPoints.first.size(); ++i)
    {
        // Rounding first of the undistorted Points -> acquivalent to the distorted Points
        const cv::Vec2d& img = caliPoints.first[i];
        const cv::Vec3d& obj = caliPoints.second[i];

        int col = static_cast<int>(std::round(img[0]));
        int row = static_cast<int>(std::round(img[1]));

        // Check inside image
        if (row < 0 || row >= mask.rows || col < 0 || col >= mask.cols)
            continue;
        if (mask.at<uchar>(row, col) == 0) {
            std::cout << "Mask was 0 at this image Points \n";
            continue;
        }

        // Direction in camera frame from undistorted normalized coordinates
        cv::Vec3d d_cam(undist[i].x, undist[i].y, 1.0);

        // Direction in world/display frame

        cv::Mat direction_world = R_inv * Cam_inv * d_cam;
        cv::Vec3d d_world(
            direction_world.at<double>(0),
            direction_world.at<double>(1),
            direction_world.at<double>(2));

        // Intersect with plane Z = Z0:
        // C.z + λ d_world.z = Z0  -> λ = (Z0 - C.z) / d_world.z
        double lambda = (Z0 - C[2]) / d_world[2];

        cv::Vec3d X = C + lambda * d_world;  // intersection point on display

        cv::Vec3d diff = X - obj;           // error on display plane

        cv::Vec2d& err = error_map.at<cv::Vec2d>(row, col);
        err = cv::Vec2d(diff[0], diff[1]);

        double norm = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
        sumsq += norm * norm;
        *maxErr = std::max(*maxErr, norm);
        ++N;
    }

    *sqrtErr = (N > 0) ? std::sqrt(sumsq / static_cast<double>(N)) : 0.0;

    return error_map;
}

cv::Mat ImageProcessing::undistortImage(const cv::Mat& img,
    const cv::Mat& K,
    const cv::Mat& distCoeffs)
{
    int W = img.cols;
    int H = img.rows;

    cv::Mat map_x(H, W, CV_32F);
    cv::Mat map_y(H, W, CV_32F);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(1, 0);
    double p1 = distCoeffs.at<double>(2, 0);
    double p2 = distCoeffs.at<double>(3, 0);
    double k3 = distCoeffs.cols >= 5 ? distCoeffs.at<double>(4, 0) : 0.0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {

            // --- Normalize in Camera Coordinates
            double x_u = (x - cx) / fx;
            double y_u = (y - cy) / fy;

            double r2 = x_u * x_u + y_u * y_u;
            double r4 = r2 * r2;
            double r6 = r4 * r2;

            double radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;

            // see lateral + radial distortion 
            double x_d = x_u * radial + 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u);
            double y_d = y_u * radial + p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u;

            // back to pixels
            map_x.at<float>(y, x) = fx * x_d + cx;
            map_y.at<float>(y, x) = fy * y_d + cy;
        }
    }

    cv::Mat img32;
    img.convertTo(img32, CV_32F);

    cv::Mat distorted;
    cv::remap(img32, distorted, map_x, map_y, cv::INTER_LINEAR);

    return distorted;
}

cv::Vec2d ImageProcessing::newtonSolverdistort(
    const cv::Vec2d& pixelCoords,   // (u_d, v_d) verzerrt in Pixeln
    const cv::Mat& cam_Matrix,
    const cv::Mat& dist_coeffs)
{
    // --- Basic sanity checks ---
    CV_Assert(cam_Matrix.type() == CV_64F);
    CV_Assert(cam_Matrix.rows == 3 && cam_Matrix.cols == 3);
    CV_Assert(dist_coeffs.type() == CV_64F);
    CV_Assert(dist_coeffs.total() >= 4);       // mind. k1, k2, p1, p2

    // --- Intrinsics ---
    const double fx = cam_Matrix.at<double>(0, 0);
    const double fy = cam_Matrix.at<double>(1, 1);
    const double cx = cam_Matrix.at<double>(0, 2);
    const double cy = cam_Matrix.at<double>(1, 2);

    // --- Distortion coefficients (k1,k2,p1,p2[,k3]) ---
    const double* dptr = dist_coeffs.ptr<double>(0);
    const double k1 = dptr[0];
    const double k2 = dptr[1];
    const double p1 = dptr[2];
    const double p2 = dptr[3];
    const double k3 = (dist_coeffs.total() > 4) ? dptr[4] : 0.0;

    // --- Distorted pixel coordinates (u_d, v_d) ---
    const double u_d = pixelCoords[0];
    const double v_d = pixelCoords[1];

    // --- Convert to normalized distorted coordinates (x_d, y_d) ---
    const double x_d = (u_d - cx) / fx;
    const double y_d = (v_d - cy) / fy;

    // --- Initial guess for undistorted normalized coordinates (x_u, y_u) ---
    double x_u = x_d;
    double y_u = y_d;

    // --- Newton iteration ---
    for (int iter = 0; iter < 5; ++iter) {
        // Radius
        const double r2 = x_u * x_u + y_u * y_u;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;

        // Radial term
        const double L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        const double dLdr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
        const double Lx = 2.0 * x_u * dLdr2;
        const double Ly = 2.0 * y_u * dLdr2;

        // Tangential distortion
        const double tx = 2.0 * p1 * x_u * y_u + p2 * (r2 + 2.0 * x_u * x_u);
        const double ty = p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * p2 * x_u * y_u;

        // Tangential derivatives
        const double dtxdx = 2.0 * p1 * y_u + 6.0 * p2 * x_u;
        const double dtxdy = 2.0 * p1 * x_u + 2.0 * p2 * y_u;
        const double dtydx = 2.0 * p1 * x_u + 2.0 * p2 * y_u;
        const double dtydy = 6.0 * p1 * y_u + 2.0 * p2 * x_u;

        // Forward distortion of current (x_u, y_u)
        const double fx = x_u * L + tx;
        const double fy = y_u * L + ty;

        // Residual: f(x_u, y_u) - (x_d, y_d) = 0
        const double R_x = fx - x_d;
        const double R_y = fy - y_d;

        // Jacobian matrix entries
        const double a = L + x_u * Lx + dtxdx;   // ∂f_x / ∂x_u
        const double b = x_u * Ly + dtxdy;       // ∂f_x / ∂y_u
        const double c = y_u * Lx + dtydx;       // ∂f_y / ∂x_u
        const double d = L + y_u * Ly + dtydy;   // ∂f_y / ∂y_u

        const double det = a * d - b * c;
        if (std::abs(det) < 1e-18) {
            break; // numerisch instabil -> abbrechen
        }

        // Newton step Δx, Δy (solve J * Δ = R)
        const double dx = (d * R_x - b * R_y) / det;
        const double dy = (-c * R_x + a * R_y) / det;

        x_u -= dx;
        y_u -= dy;

        if (dx * dx + dy * dy < 1e-18) {
            break; // konvergiert
        }
    }

    // --- Back to pixel coordinates (u_undist, v_undist) ---
    const double u_undist = fx * x_u + cx;
    const double v_undist = fy * y_u + cy;

    return cv::Vec2d(u_undist, v_undist);
}



std::vector<cv::Mat> ImageProcessing::do_wrapped_Phase(const std::vector<cv::Mat>& vec,
    int n_pics_perPhase,
    int n_shifts)
{
    CV_Assert(!vec.empty());
    CV_Assert(vec[0].channels() == 1);

    // --- Build mean images ---

    std::vector<cv::Mat> mean_vec;
    mean_vec.reserve(vec.size());

    if (n_pics_perPhase > 1) {
        auto it = vec.begin();
        while (it != vec.end()) {

            auto it2 = it + n_pics_perPhase;
            cv::Mat meanImg = mean(std::vector<cv::Mat>(it, it2));
            meanImg.convertTo(meanImg, CV_64F);

            mean_vec.push_back(meanImg);
            it = it2;
        }
    }
    else {
        for (const auto& f : vec) {
            cv::Mat tmp;
            f.convertTo(tmp, CV_64F);
            mean_vec.push_back(tmp);
        }
    }

    // --- Allocate S1,S2,S3 ---

    cv::Size size = mean_vec[0].size();

    std::array<cv::Mat, 2> s1, s2, s3;
    for (int p = 0; p < 2; ++p) {
        s1[p] = cv::Mat::zeros(size, CV_64F);
        s2[p] = cv::Mat::zeros(size, CV_64F);
        s3[p] = cv::Mat::zeros(size, CV_64F);
    }

    // --- 3) Precompute sin/cos LUT ---
    std::vector<double> sines(n_shifts), cosines(n_shifts);
    for (int i = 0; i < n_shifts; ++i) {
        double ph = double((CV_2PI * i) / n_shifts);
        sines[i] = std::sin(ph);
        cosines[i] = std::cos(ph);
    }

    // --- 4) Accumulate S1, S2, S3 ---
    for (int i = 0; i < n_shifts; ++i) {

        s1[0] += mean_vec[i] * sines[i];
        s2[0] += mean_vec[i] * cosines[i];
        s3[0] += mean_vec[i];

        s1[1] += mean_vec[i + n_shifts] * sines[i];
        s2[1] += mean_vec[i + n_shifts] * cosines[i];
        s3[1] += mean_vec[i + n_shifts];
    }

    // --- 5) Compute wrapped phase and contrast---

    std::vector<cv::Mat> wrapped(2), contrast(2), baseIntensity(2);
    
    for (int p = 0; p < 2; ++p) {

        wrapped[p] = cv::Mat(size, CV_64F);
        contrast[p] = cv::Mat(size, CV_64F);
        baseIntensity[p] = s3[p] / n_shifts;
        cv::Mat mag(size, CV_64F);

        cv::parallel_for_(cv::Range(0, size.height),
            [&](const cv::Range& r) {
                for (int y = r.start; y < r.end; ++y) {

                    const double* s1p = s1[p].ptr<double>(y);
                    const double* s2p = s2[p].ptr<double>(y);
                    const double* s3p = s3[p].ptr<double>(y);

                    double* wp = wrapped[p].ptr<double>(y);
                    double* cp = contrast[p].ptr<double>(y);
                    double* mp = mag.ptr<double>(y);

                    for (int x = 0; x < size.width; ++x) {

                        double a = s1p[x];
                        double b = s2p[x];
                        double c = s3p[x];

                        wp[x] = std::atan2(-a, b);
                        mp[x] = std::sqrt(a * a + b * b);
                        cp[x] = (2.0f * mp[x]) / c;
                    }
                }
            });
    }
    std::vector<cv::Mat> return_container;
    for (auto& m : wrapped) { return_container.push_back(std::move(m)); }
    for (auto& m : contrast) { return_container.push_back(std::move(m)); }
    for (auto& m : baseIntensity) { return_container.push_back(std::move(m)); }
    return return_container;
}

void ImageProcessing::unwrap_row(const cv::Mat& wrapped, cv::Mat& unwrapped, const cv::Mat& mask, int row)
{
    double two_pi = CV_2PI;
    /*normalizeAndDisplay(wrapped);
    normalizeAndDisplay(mask);*/
    double prev = 0;
        
    double k = 0;

    int valid_counter{0};

    cv::Mat wrapped64;
    wrapped.convertTo(wrapped64, CV_64F);

    for (int col = 0; col < wrapped.cols; ++col)
    {
        if (!mask.at<uchar>(row, col))
            continue;
        
        double current = wrapped64.at<double>(row, col);
        //Bei dem 1. validen Wert macht eine differenzbildung noch keinen Sinn
        if (!valid_counter) {
            unwrapped.at<double>(row, col) = current;
            prev = current;
            valid_counter++;
            continue;
        }
        double diff = current - prev;

        if (diff > CV_PI)      k -= 1;
        else if (diff < -CV_PI) k += 1;

        unwrapped.at<double>(row, col) = current + k * two_pi;

        prev = current;
        valid_counter++;
    }
}

void ImageProcessing::unwrap_column(const cv::Mat& wrapped, cv::Mat& unwrapped, const cv::Mat& mask, int col)
{
    const double two_pi = CV_2PI;

    double prev = 0.0;
    double k = 0.0;
    int valid_counter = 0;

    for (int row = 0; row < wrapped.rows; ++row)
    {
        if (!mask.at<uchar>(row, col))
            continue;

        double current = wrapped.at<double>(row, col);

        // erster gültiger Pixel in der Spalte
        if (!valid_counter)
        {
            unwrapped.at<double>(row, col) = current;
            prev = current;      
            valid_counter++;
            continue;
        }

        double diff = current - prev;

        // wrap detection
        if (diff > CV_PI)
            k -= 1;
        else if (diff < -CV_PI)
            k += 1;

        double unwrapped_val = current + k * two_pi;

        unwrapped.at<double>(row, col) = unwrapped_val;

        prev = current;

        valid_counter++;
    }
}

std::vector<cv::Mat> ImageProcessing::manual_phaseUnwrap(
    const std::vector<cv::Mat>& wrapped,
    const std::vector<cv::Mat>& contrast) 
{
    CV_Assert(!wrapped.empty() && !contrast.empty());
    CV_Assert(wrapped[0].type() == contrast[0].type());
    CV_Assert(wrapped[0].size() == contrast[0].size());

    cv::Mat mask = createMask(contrast, 0.5);

    std::vector<cv::Mat> unwrapped_phase(2);
    for (auto& m : unwrapped_phase) { m = cv::Mat::zeros(wrapped[0].size(), CV_64F); }
    
    for (std::size_t count = 0; count < wrapped.size(); count++) {

        if (count == 0)
        {
            // horizontal unwrap
            for (int row = 0; row < wrapped[0].rows; ++row)
                unwrap_row(wrapped[0], unwrapped_phase[count], mask, row);
        }
        else if (count == 1)
        {
            // vertical unwrap
            for (int col = 0; col < wrapped[1].cols; ++col)
                unwrap_column(wrapped[1], unwrapped_phase[count], mask, col);
        }
    }
   
    /*for (auto& img : unwrapped_phase) {
        img.setTo(cv::Scalar(0), ~mask);
    }*/
    /*normalizeAndDisplay(wrapped[0]);
    normalizeAndDisplay(wrapped[1]);

    normalizeAndDisplay(unwrapped_phase[0]);
    normalizeAndDisplay(unwrapped_phase[1]);*/
    
    return unwrapped_phase;
}


cv::Mat ImageProcessing::createMask(
    const std::vector<cv::Mat>& vec,
    double threshold) 
{
    cv::Mat maskbin, sum_contrast_n;
    CV_Assert(m_contrast[0].type() == CV_64F);
    CV_Assert(vec.size() == 2);

    if (threshold == 0.0) {
        return cv::Mat::ones(vec[0].size(), CV_8U);
    }

    cv::Mat sum_contrast = (vec[0] + vec[1])/2;
    // minMaxloc data for sum_contrast
    cv::Mat mask_8u;
    cv::normalize(sum_contrast, mask_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    minmaxloc data{ get_minmaxloc(mask_8u) };
    // Threshold

    cv::threshold(mask_8u, maskbin, threshold * data.maxval, 1, cv::THRESH_BINARY);
    //normalizeAndDisplay(mask);
    // Create Structuring Element for opening&closing
    cv::Mat strucutre = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    cv::morphologyEx(maskbin, maskbin, cv::MORPH_OPEN, strucutre, cv::Point2d(-1, -1), 2);
    cv::morphologyEx(maskbin, maskbin, cv::MORPH_CLOSE, strucutre, cv::Point2d(-1, -1), 2);

    //normalizeAndDisplay(maskbin);
    return maskbin;
}




// Function that takes two images. If Mask image is != 0 the original value is saved. Else it is 0.
// When the double value is != 0 all values that not masked are shifted about this value. 
cv::Mat ImageProcessing::applyMask(const cv::Mat& mask, const cv::Mat& img, float shift) {
    assert(mask.size() == img.size() && "Mask and image must be of the same size \n");
    assert((img.type() == CV_32FC1 || img.type() == CV_64F) && "Expected float image");

    cv::Mat result;

    switch (static_cast<int>(current_type)) {
    case 1: { // float_t
        result = cv::Mat::zeros(img.size(), CV_32FC1);
        for (int row = 0; row < img.rows; ++row) {
            const float* pImg = img.ptr<float>(row);
            const uchar* pMask = mask.ptr<uchar>(row);
            float* pRes = result.ptr<float>(row);
            if (!shift) {
                for (int col = 0; col < img.cols; ++col)
                    pRes[col] = (pMask[col] != 0.0f) ? pImg[col] : 0.0f;
            }
            else {
                for (int col = 0; col < img.cols; ++col)
                    pRes[col] = (pMask[col] != 0.0f) ? (pImg[col] + shift) : 0.0f;
            }
        }
        break;
    }
    case 2: { // double_t
        result = cv::Mat::zeros(img.size(), CV_64FC1);
        for (int row = 0; row < img.rows; ++row) {
            const double* pImg = img.ptr<double>(row);
            const uchar* pMask = mask.ptr<uchar>(row);
            double* pRes = result.ptr<double>(row);
            if (!shift) {
                for (int col = 0; col < img.cols; ++col)
                    pRes[col] = (pMask[col] != 0.0) ? pImg[col] : 0.0;
            }
            else {
                for (int col = 0; col < img.cols; ++col)
                    pRes[col] = (pMask[col] != 0.0) ? (pImg[col] + shift) : 0.0;
            }
        }
        break;
    }
    default:
        throw std::runtime_error("Unknown current_type in applyMask()");
    }

    //normalizeAndDisplay(result);
    return result;
}

// Shifts all valid pixels by the given shift value (IMG(x,y) -= shift_value).
// Sets all invalid (masked-out) pixels to -10.0 (float) or -10.0 (double).
void ImageProcessing::shiftStartPhasetoZero(const cv::Mat& mask, cv::Mat& img, float shift_value) {
    assert(mask.size() == img.size());
    assert((img.type() == CV_32FC1 || img.type() == CV_64FC1) && "Expected float or double image");

    switch (static_cast<int>(current_type)) {
    case 1: { // float_t
        for (int row = 0; row < img.rows; ++row) {
            float* pImg = img.ptr<float>(row);
            const uchar* pMask = mask.ptr<uchar>(row);

            for (int col = 0; col < img.cols; ++col) {
                if (pMask[col] == 0.0f)
                    pImg[col] = -10.0f;
                else
                    pImg[col] -= shift_value;
            }
        }
        break;
    }

    case 2: { // double_t
        for (int row = 0; row < img.rows; ++row) {
            double* pImg = img.ptr<double>(row);
            const uchar* pMask = mask.ptr<uchar>(row);

            for (int col = 0; col < img.cols; ++col) {
                if (pMask[col] == 0.0)
                    pImg[col] = -10.0;
                else
                    pImg[col] -= static_cast<double>(shift_value);
            }
        }
        break;
    }

    default:
        throw std::runtime_error("Unknown current_type in shiftStartPhasetoZero()");
    }
}


//
////Just a small example code to do generic programming with iterators
//template <class InputIt>
//typename std::iterator_traits<InputIt>::difference_type
//distance(InputIt first, InputIt last) {
//    using category = typename std::iterator_traits<InputIt>::iterator_category;
//
//    if constexpr (std::is_same_v<category, std::random_access_iterator_tag>)
//        return last - first;     // fast O(1)
//    else {
//        typename std::iterator_traits<InputIt>::difference_type n = 0;
//        for (; first != last; ++first) ++n;   // slow O(n)
//        return n;
//    }
//}


//typename std::iterator_traits<iterator>::iterator_category 
// I first had the above as datatype. but the return type is of type std::iterator_traits<...>::iterator_category
template<typename iterator>
iterator advance_return(iterator start, int n) {
    return start + n;
}

//does copy the iterator, Therefore the original does not get changed. 
template<typename iterator>
constexpr iterator advance_return1(iterator start, typename std::iterator_traits<iterator>::difference_type n) {
    // This line does not work. STL templates mostly do not allow excplicit templeta args becasue it leads to ambigous function deduction.
    // std::advance<iterator, typename std::iterator_traits<iterator>::difference_type>(start, n);
    std::advance(start, n);
    return start;
}


std::vector<std::pair<double,double>> 
ImageProcessing::gray_value_calib(const std::vector<cv::Mat>& vec, int pics_per_val, int stepwidth) {
    // Create first mean values if needed. 
    CV_Assert(!vec.empty() && "If the vector is empty there must be error in acquisition. \n");
    CV_Assert(vec[0].channels() == 1);
    std::vector<cv::Mat> mean_vec{};
    std::vector<std::pair<double, double>> LUT{};
    // Build the mean value over pics_per_val
    if (pics_per_val > 1) {
        std::vector<cv::Mat>::const_iterator begin_mean = vec.cbegin();
        
        std::vector<cv::Mat>::const_iterator end_guard = vec.end();

        for (std::size_t i = 0; i < (std::numeric_limits<uchar>::max() / stepwidth); ++i) {
            std::vector<cv::Mat>::const_iterator end_mean = advance_return1(begin_mean, pics_per_val);
            CV_Assert(end_mean <= end_guard && "Iterator dereference element out of bounds");
            
            mean_vec.push_back(mean(std::vector<cv::Mat>(begin_mean, end_mean)));
            begin_mean = end_mean;
        }
    }
    else {
        for (const auto& f : vec)
        {
            cv::Mat tmp;
            f.convertTo(tmp, CV_64F);
            mean_vec.push_back(tmp);
        }
    }
    
    CV_Assert(mean_vec.size() >= 2 && "Grayvalue calib requires min. 2 values");

    // Subtrakt the first image (black) from the brightest (white) the difference is hopefully a usable mask;
    cv::Mat mask = mean_vec.back() - mean_vec.front();
    cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX, CV_8U);
    minmaxloc mask_info{ get_minmaxloc(mask) };
    cv::threshold(mask, mask, mask_info.maxval * 0.5, 255, cv::THRESH_BINARY);
    normalizeAndDisplay(mask);

    std::vector<cv::Scalar_<double>> mean_values;

    for (std::size_t i = 0; i < mean_vec.size(); ++i) {
        double GTgray_val = i * stepwidth;
        double meassure_gray_val = cv::norm(cv::mean(mean_vec[i], mask));
        LUT.push_back(std::pair<double, double>(GTgray_val, meassure_gray_val));
    }


    // --- This part old. Should be DELETED (save pictures within class) 
    for (const auto& mean_img : mean_vec) {
        mean_values.push_back(cv::mean(mean_img, mask));
    }
    m_mean_grayValues = std::move(mean_values);

    return LUT;
}


/*
// This is the first implementation of a template elipsis. The function takes a arbitrary ammount of cv::MAts and a function pointer. 
// Becasue the cv::Mat datatype can not be decued at compile time this functions template calls a second inner function where, 
// for the datatypes it is differntiatied. 
template<typename func, typename... Mats>
auto ImageProcessing::forEachPixel(func, Mats&&... mats) {
    static_assert(sizeof...(mats) > 0, "Need at least one matrix."); //compile time check
    auto first = std::get<0>(std::forward_as_tuple(std::forward<Mats>(mats)));
    const cv::Size size = first.size();
    const int type = first.type();
    (assert(size == mats.size() && type == mats.type()), ...); //compiler expands at compile time, checkupt at runtime.
    // giving a conditional datatype back, std::is_same<>::value is static function
    // if condition is true give back the first datatype, if false give back the second datatype 
    // But mats.size() and mats.type() are runtime functions, therefore static assert does not work. 
    
    // This line does not work becasue in decltype it is assumed that .at<float> for each datatype
    // This line typename std::conditional -> typename is necessary to tell the compiler that a type is named. not a value. wihtout std::conditional<...>::type could be interpreted as static
    // typenmae std::remove_reference -> the function itself is template class and ::type nested type alias. 
    // Whenever <T>::type (or something) is used and it is not a static member function we have to use typename becasue the comiler can not now. 
    // using T = typename std::conditional<
    //    std::is_same<Func, float(*)(float,float) >>::value, float, typename std::remove_reference<decltype(first.at<float>(0, 0))>::type>::type;
    
    // typenabhängiger Name. 
    int type = mat.type();           
    int depth = CV_MAT_DEPTH(type);  //Macros to extract depth (datatype)
    int channels = CV_MAT_CN(type);  //Macros to define extract how manc channels are there
    
    // type dependend name here, therefore "typename" before. Also this line does not work. becasue type is runtime constant at the 
    // template deduction would hapen at comile time the value is not available when the programm runs. 
    //using T = typename CvDepthTraits<CV_MAT_DEPTH(type)>::value_type;

    switch (depth) {
    case CV_8U:  return forEachPixelImpl<CV_8U>(func, std::forward<Mats>(mats)...);
    case CV_8S:  return forEachPixelImpl<CV_8S>(func, std::forward<Mats>(mats)...);
    case CV_16U: return forEachPixelImpl<CV_16U>(func, std::forward<Mats>(mats)...);
    case CV_16S: return forEachPixelImpl<CV_16S>(func, std::forward<Mats>(mats)...);
    case CV_32S: return forEachPixelImpl<CV_32S>(func, std::forward<Mats>(mats)...);
    case CV_32F: return forEachPixelImpl<CV_32F>(func, std::forward<Mats>(mats)...);
    case CV_64F: return forEachPixelImpl<CV_64F>(func, std::forward<Mats>(mats)...);
    default:
        throw std::runtime_error("Unsupported depth.");
    }
}


template<int Depth, typename Func, typename... Mats>
cv::Mat ImageProcessing::forEachPixelImpl(const Func& func, Mats&&... mats) {
    using T = typename CvDepthTraits<Depth>::value_type;   //again typename necessary because of ...<dependen>::...
    auto&& first = std::get<0>(std::forward_as_tuple(mats...));

    cv::Mat result(first.size(), first.type());

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            result.at<T>(y, x) = func(mats.at<T>(y, x)...);
        }
    }
    return result;
}
*/


////Try to write first allcoator
//template <typename T, std::size_t Alignment = 32>
//struct AlignedAllocator {
//    using value_type = T;
//
//    AlignedAllocator() noexcept = default;
//    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
//
//    T* allocate(std::size_t n) {
//        void* ptr = nullptr;
//        if (ptr = std::aligned_alloc() != 0)
//            throw std::bad_alloc();
//        return reinterpret_cast<T*>(ptr);
//    }
//
//    void deallocate(T* p, std::size_t) noexcept {
//        free(p);
//    }
//};

template <typename T>
using decay_all_t = typename std::remove_cv<typename std::remove_reference<typename std::remove_pointer<T>::type>::type>::type;


void ImageProcessing::calc_reproject_error(bool visualizing) {
    assert(runtime_flags.disp.wavelength && "Parameters of the phase pattern are not available. Call generatePattern() before \n");
    
    //m_mask = { createMask() };
    cv::Mat unwrap1_masked, unwrap2_masked;
    
    unwrap1_masked = applyMask(m_mask, m_unwrapped_phase[0]);
    unwrap2_masked = applyMask(m_mask, m_unwrapped_phase[1]);
    minmaxloc masked_unwrap1{ get_minmaxloc(unwrap1_masked) };
    minmaxloc masked_unwrap2{ get_minmaxloc(unwrap2_masked) };
    std::cout << "Horizontal unwrap " << masked_unwrap1 << '\n' <<
        "Vertical unwrap " << masked_unwrap2 << '\n';

    //Not valid pixels at -10 
    shiftStartPhasetoZero(m_mask, unwrap1_masked, masked_unwrap1.minval);
    shiftStartPhasetoZero(m_mask, unwrap2_masked, masked_unwrap2.minval);

    switch (static_cast<int>(current_type)) {
    case(1): {
        //auto calibPoints1 = generateCalibrationPoints<float>(unwrap1_masked, unwrap2_masked, runtime_flags.disp.wavelength);
        auto caliPoints = generateCalibrationPoints<float>(unwrap1_masked, unwrap2_masked, runtime_flags.disp.wavelength,
            runtime_flags.camera_data.pixel_x, runtime_flags.camera_data.pixel_y);
        cv::Mat rvec, tvec;
        bool ok = cv::solvePnP(caliPoints.objectPoints, caliPoints.imagePoints,
                m_calib_data.cameraMatrix, m_calib_data.distCoeffs, rvec, tvec,
                false, 
                cv::SOLVEPNP_ITERATIVE);   
        if (!ok) throw std::runtime_error("solvePnP failed.");
        std::cout << "Translation vec " << cv::norm(tvec) << '\n';
        

        //auto repro2to3error = project2to3d(caliPoints, m_calib_data.cameraMatrix,
        //    m_calib_data.distCoeffs, rvec, tvec);

        std::cout << "Caluculated the float path \n";
        // Stored as std::variants< ... <float>, ... <double>>
        m_calib_points = caliPoints;
        //m_repro_error = repro2to3error;
        break;
    }
    case(2): {
        //std::cout << "float " << CV_32F << "\n" << "double" << CV_64F << '\n' <<
        //    "unwrap1_masked: " << unwrap1_masked.type() << '\n' <<
        //    "unwrap2_masked: " << unwrap2_masked.type() << '\n';
        //auto calibPoints1 = generateCalibrationPoints<double>(unwrap1_masked, unwrap2_masked, runtime_flags.disp.wavelength);
        auto caliPoints = generateCalibrationPoints<double>(unwrap1_masked, unwrap2_masked, runtime_flags.disp.wavelength,
            runtime_flags.camera_data.pixel_x, runtime_flags.camera_data.pixel_y);
        cv::Mat rvec, tvec;
        // get types
        auto* ptr1 = caliPoints.imagePoints.data();
        using type1 = decay_all_t<decltype(ptr1)>;
        //typeid() return a std::typeinfo object. This has method .name()
        std::cerr << "Type of image Points after decay " << typeid(type1).name() << '\n';

        auto* ptr2 = caliPoints.objectPoints.data();
        using type2 = decay_all_t<decltype(ptr2)>;
        std::cerr << "Type of object points after decay " << typeid(type2).name() << '\n';
        std::cout << "Size obejct points " << caliPoints.objectPoints.size() << '\n' <<
            "size image point " << caliPoints.imagePoints.size() << '\n';

        bool ok = cv::solvePnP(caliPoints.objectPoints, caliPoints.imagePoints,
            m_calib_data.cameraMatrix, m_calib_data.distCoeffs, rvec, tvec,
            false,
            cv::SOLVEPNP_ITERATIVE);
        if (!ok) throw std::runtime_error("solvePnP failed.");
        std::cout << "Translation vec " << cv::norm(tvec) << '\n';
        
        //auto repro2to3error = project2to3d(caliPoints, m_calib_data.cameraMatrix,
        //    m_calib_data.distCoeffs, rvec, tvec);

        std::cout << "Caluculated the double path \n";
        // Stored as std::variants< ... <float>, ... <double>>
        m_calib_points = caliPoints;
        //m_repro_error = repro2to3error;
        break;
    }
    default:
        throw std::runtime_error("Unknown current_type in shiftStartPhasetoZero()");
    }
}


void ImageProcessing::save_Reprodata(const std::string& path) {
    try {
        //if (std::holds_alternative<ReprojectionError<double>>(m_repro_error)) {
        //    auto& repro_error_typed{ std::get<ReprojectionError<double>>(&m_repro_error) };
        //    repro_error_typed -> 
        //}

        auto* ptr1 = std::get_if<ReprojectionError<double>>(&m_repro_error);
        auto* ptr2 = std::get_if<ReprojectionError<float>>(&m_repro_error);
        std::string final(path + "/ReprojectionError.xml");
        std::filesystem::path final_path(final);
        if (std::filesystem::exists(final_path)) {
            std::cout << "File for reprojection Error does already exist \n" << "Possible override. Do you want to proceed? \n [Y/N]";
            char c;
            bool cond{ true };
            do {
                std::cin >> c;
                if (!std::cin.fail()) {
                    std::cin.clear();
                    std::cin.ignore(100, '\n');
                }
                if (std::toupper(c) == 'N') {
                    return;
                }
                if (std::toupper(c) == 'Y') {
                    cond = false;
                }
            } while (cond);
        }

        if (ptr1) {
            cv::FileStorage fs(final, cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs.write(m_reprojection_error_string[0], ptr1->sqrt_Error_x);
                fs.write(m_reprojection_error_string[1], ptr1->sqrt_Error_y);
                fs.write(m_reprojection_error_string[2], ptr1->median_x);
                fs.write(m_reprojection_error_string[3], ptr1->median_y);
                fs.write(m_reprojection_error_string[4], ptr1->max_Error_x);
                fs.write(m_reprojection_error_string[5], ptr1->max_Error_y);
                //Here also save the median of the error.
            }
        }
        else if (ptr2) {
            cv::FileStorage fs(final, cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs.write(m_reprojection_error_string[0], ptr2->sqrt_Error_x);
                fs.write(m_reprojection_error_string[1], ptr2->sqrt_Error_y);
                fs.write(m_reprojection_error_string[2], ptr2->median_x);
                fs.write(m_reprojection_error_string[3], ptr2->median_y);
                fs.write(m_reprojection_error_string[4], ptr2->max_Error_x);
                fs.write(m_reprojection_error_string[5], ptr2->max_Error_y);
                //Here also save the median of the error.
            }
        }
    }
    catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; }
}


void ImageProcessing::load_calib(std::string path) {
    m_calib_data = getfromFile(path);
}


// carefull && binds tighter than ||
void ImageProcessing::load_frames(const std::string& path) {
    try {
        //Checks only for the first element in arrays. It is assumed when the first is empty the second one must be to. 
        if ((m_baseIntensity[0].empty() || cv::norm(cv::sum(m_baseIntensity[0])) == 0.0) &&
            (m_contrast[0].empty() || cv::norm(cv::sum(m_contrast[0])) == 0.0) &&
            (m_wrapped_phase[0].empty() || cv::norm(cv::sum(m_wrapped_phase[0])) == 0.0) &&
            (m_unwrapped_phase[0].empty() || cv::norm(cv::sum(m_unwrapped_phase[0])) == 0.0))
        {
            //create();
            cv::FileStorage fs(path, cv::FileStorage::READ);
            if (fs.isOpened()) {
                fs[m_save_keys[0]] >> m_wrapped_phase[0];
                fs[m_save_keys[1]] >> m_wrapped_phase[1];
                fs[m_save_keys[2]] >> m_contrast[0];
                fs[m_save_keys[3]] >> m_contrast[1];
                fs[m_save_keys[4]] >> m_baseIntensity[0];
                fs[m_save_keys[5]] >> m_baseIntensity[1];
                fs[m_save_keys[6]] >> m_unwrapped_phase[0];
                fs[m_save_keys[7]] >> m_unwrapped_phase[1];
            }
            fs.release();
            //normalizeAndDisplay(m_wrapped_phase[0]);
        }
        else { throw std::runtime_error("The image container already have data in it. This is not allowed. \n"); }

    }
    catch (std::exception& e) { std::cout << "EXCEPTION " << e.what(); }
}


void ImageProcessing::create() {
    for (auto& m : m_s1) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_s2) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_s3) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_baseIntensity) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_contrast) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_wrapped_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_unwrapped_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
}

// create() initializes the arrays for the processing. This methods needs to be called one time before further processsing. 
void ImageProcessing::create(std::vector<cv::Mat>& vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            std::cerr << "All pictures must be same kind and type. \n";
            throw std::runtime_error("All pictures must be same kind and type. \n");
        }
    }
    for (auto& m : m_s1) m = cv::Mat(vec.at(0).size(), CV_32F, cv::Scalar(0.0));
    for (auto& m : m_s2) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_s3) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_baseIntensity) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_contrast) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_wrapped_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_unwrapped_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
}

void ImageProcessing::bayerToGray()
{
    for (const auto& frame : m_phase) {
		cv::cvtColor(frame, frame, cv::COLOR_BayerRG2GRAY);
    }
}

auto normalize = [](const cv::Mat& pic) -> cv::Mat_<uchar> {
    if (pic.type() == CV_8U) return pic;
    cv::Mat_<uchar> dest;
    cv::normalize(pic, dest, 0, 255, cv::NORM_MINMAX, CV_8U);
    return dest;
    };

/*
auto save_png = [](const std::filesystem::path& path, const std::vector<cv::Mat>& vec) -> void {
    for (std::size_t i{ 0 }; i < vec.size(); ++i) {
        cv::imwrite((path / std::to_string(i) / ".png").string(), normalize(vec[i]));
    }
    };

auto save_png = [](const std::filesystem::path& path, const std::array<cv::Mat, 2>& arr) -> void {
    for (std::size_t i{ 0 }; i < arr.size(); ++i) {
        cv::imwrite((path / std::to_string(i) / ".png").string(), normalize(arr[i]));
    }
    };
*/


void ImageProcessing::saveImages_png(const std::string& path_s) {
    std::filesystem::path path{ path_s };
    if (!std::filesystem::exists(path)) {
        if (!std::filesystem::create_directory(path)) std::cerr << "Creating the directory failed. \n";
        std::cout << "Created Directory for the .png files \n";
    }
    else {
        std::cout << "Directory for .png files already exists \n" << "Possible override. Do you want to proceed? \n [Y/N]";
        char c;
        bool cond{ true };
        do {
            std::cin >> c;
            if (!std::cin.fail()) {
                std::cin.clear();
                std::cin.ignore(100, '\n');
            }
            if (std::toupper(c) == 'N') {
                return;
            }
            if (std::toupper(c) == 'Y') {
                cond = false;
            }
        } while (cond);
    }
    //Create save structure
    std::array<std::string, 7> strings{"Full_frames", "Mean_frames", "Base Intensity", "Contrast",
        "WrappedPhase", "unwrappedPhase", "reprojectionError"};

    for (std::size_t i = 0; i < strings.size(); ++i)
    {
        if (!std::filesystem::create_directory(path / strings[i])) {
            std::cout << "Creating directory " << (path / strings[i]).string() << " failed \n";
        }
        switch (i) {
        case(0): { save_png(path / strings[i], m_frames); break; }
        case(1): { save_png(path / strings[i], m_raw_phase); break; }
        case(2): { save_png(path / strings[i], m_baseIntensity); break; }
        case(3): { save_png(path / strings[i], m_contrast); break; }
        case(4): { save_png(path / strings[i], m_wrapped_phase); break; }
        case(5): { save_png(path / strings[i], m_unwrapped_phase); break; }
        case(6): { save_png(path / strings[i], m_reprojection_error_img); break; }
            
        }
    }
}


void ImageProcessing::saveImages(const std::string& path) {
    std::filesystem::path complete(path);
    if (std::filesystem::exists(complete)) {
        throw std::runtime_error("Directory already exist. Data will be overriden. \n");
    }
    try {
        std::filesystem::create_directories(complete);
    }
    catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl; }

    std::cout << "Created directory at " << complete.string() << '\n';

    cv::FileStorage fs((complete/"Data.xml").string(), cv::FileStorage::WRITE);
    
    if (fs.isOpened()) {
        fs << "wrappedPhasehorizontal" << m_wrapped_phase.at(0);
        fs << "wrappedPhasevertical" << m_wrapped_phase.at(1);
        fs << "contrasthorizontal" << m_contrast.at(0);
        fs << "contrastvertical" << m_contrast.at(1);
        fs << "baseIntensityhorizontal" << m_baseIntensity.at(0);
        fs << "baseIntensityvertical" << m_baseIntensity.at(1);
        fs << "unwrappedPhasehorizontal" << m_unwrapped_phase.at(0);
        fs << "unwrappedPhasevertical" << m_unwrapped_phase.at(1);
        fs << "reprojectionErrorHorizontal" << m_reprojection_error.at(0);
        fs << "reprojectionErrorVertical" << m_reprojection_error.at(1);
    }
}

std::vector<double> ImageProcessing::extract_Row_reprojection(int row) {
    const cv::Mat& repro = m_reprojection_error[0];
    std::vector<double> row_repro;
    row_repro.reserve(repro.cols);
    for (int i = 0; i < repro.cols; ++i) {
        if (!m_mask.at<uchar>(row, i)) continue;
        row_repro.push_back(repro.at<double>(row, i));
    }
    return row_repro;
}

std::vector<double> ImageProcessing::extract_Row_unwrap(int row) {
    const cv::Mat& repro = m_unwrapped_phase[0];
    std::vector<double> row_unwrap;
    row_unwrap.reserve(repro.cols);
    for (int i = 0; i < repro.cols; ++i) {
        if (!m_mask.at<uchar>(row, i)) continue;
        row_unwrap.push_back(repro.at<double>(row, i));
    }
    return row_unwrap;
}

std::vector<double> ImageProcessing::extract_Column_reprojection(int x) {
    const cv::Mat& repro = m_reprojection_error[1];
    std::vector<double> col;
    col.reserve(repro.rows);

    for (int i = 0; i < repro.rows; ++i) {
        if (!m_mask.at<uchar>(i, x)) continue;
        col.push_back(repro.at<double>(i, x));
    }
    return col;
}

std::vector<double> ImageProcessing::extract_Column_unwrap(int x) {
    const cv::Mat& unwrap = m_unwrapped_phase[1];
    std::vector<double> col;
    col.reserve(unwrap.rows);

    for (int i = 0; i < unwrap.rows; ++i) {
        if (!m_mask.at<uchar>(i, x)) continue;
        col.push_back(unwrap.at<double>(i, x));
    }
    return col;
}

ImageProcessing::~ImageProcessing() {
    --instance_counter;
}


void ImageProcessing::goldsteinUnwrap() {
    cv::Mat unwrapped1;
    cv::Mat unwrapped2;
    for (auto& m : m_unwrapped_phase) {
		cv::normalize(m, m, 0, 1, cv::NORM_MINMAX, CV_32F); 
    }
    cv::Mat mask1 = (m_contrast[0] > 0.2f); // for example: keep only valid contrast regions
    mask1.convertTo(mask1, CV_8U);           // ensure binary mask
	cv::Mat mask2 = (m_contrast[1] > 0.2f);
	mask2.convertTo(mask2, CV_8U);
    goldsteinUnwrapCV(m_wrapped_phase[0], unwrapped1, mask1);
	goldsteinUnwrapCV(m_wrapped_phase[1], unwrapped2, mask2);

	cv::normalize(unwrapped1, m_unwrapped_phase[0], 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::normalize(unwrapped2, m_unwrapped_phase[1], 0, 255, cv::NORM_MINMAX, CV_8U);    
	cv::imshow("Goldstein Unwrapped Phase 1", m_unwrapped_phase.at(0));
	cv::imshow("Goldstein Unwrapped Phase 2", m_unwrapped_phase.at(1));
    cv::waitKey(0);

}


std::vector<cv::Mat> ImageProcessing::unwrapped_phase(
    const std::vector<cv::Mat>& wrappedPhase,
    const std::vector<cv::Mat>& contrast) 
{
    CV_Assert(wrappedPhase.size() == 2);
    CV_Assert(contrast.size() == 2);
    CV_Assert(wrappedPhase[0].size() == contrast[0].size());
    CV_Assert(wrappedPhase[0].type() == CV_32F || wrappedPhase[0].type() == CV_64F);

    
    cv::Size sz = wrappedPhase[0].size();

    cv::Mat wX, wY, cX, cY;
    wrappedPhase[0].convertTo(wX, CV_32F);
    wrappedPhase[1].convertTo(wY, CV_32F);

    cv::Mat mask = { createMask(contrast, 0.5) };
    //normalizeAndDisplay(wX);
    std::vector<cv::Mat> unwrapped(2);
    unwrapped[0] = cv::Mat(sz, CV_64F);
    unwrapped[1] = cv::Mat(sz, CV_64F);
    cv::Mat unwrapX;
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params params;
    params.width = sz.width;
    params.height = sz.height;
    params.histThresh = CV_PI / 10;
    params.nbrOfSmallBins = 20;
    params.nbrOfLargeBins = 10;

    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> unwrapXalgo =
        cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);

    unwrapXalgo->unwrapPhaseMap(wX, unwrapX, mask);
    unwrapX.convertTo(unwrapped[0], CV_64F);
    normalizeAndDisplay(unwrapX);
    //normalizeAndDisplay(wY);
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params paramsY;
    paramsY.width = wrappedPhase[0].rows;
    paramsY.height = wrappedPhase[0].cols;
    paramsY.histThresh = CV_PI / 10;
    paramsY.nbrOfSmallBins = 20;
    paramsY.nbrOfLargeBins = 10;

    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> unwrapYalgo =
        cv::phase_unwrapping::HistogramPhaseUnwrapping::create(paramsY);
    cv::Mat unwrap_t;
    cv::Mat wY_t = wY.t();
    cv::Mat mask_t = mask.t();
    unwrapYalgo->unwrapPhaseMap(wY_t, unwrap_t, mask_t);

    cv::Mat unwrap = unwrap_t.t();
    unwrap.convertTo(unwrapped[1], CV_64F);
    normalizeAndDisplay(unwrapped[1]);
    return unwrapped;
}

// Converts every Img to CV64F. Take the man value of the vector and return a CV64F img. 
cv::Mat ImageProcessing::mean(const std::vector<cv::Mat>& vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            throw std::runtime_error ("All pictures must be same kind and type. \n");
        }
    }
    cv::Mat acc;

    if(vec[0].type() != CV_64F) vec[0].convertTo(acc, CV_64FC1);
    
    for (size_t i = 1; i < vec.size(); ++i) {
        cv::Mat temp;
        vec[i].convertTo(temp, CV_64FC1);
        acc += temp;   // pixelweise Addition
    }

    acc /= static_cast<double>(vec.size());  // pixelweise Division

    return acc;
}


cv::Mat ImageProcessing::load_images(std::string path) {
    std::filesystem::path image_location{ path };
    if (std::filesystem::exists(image_location)) {
        return cv::imread(path, cv::ImreadModes::IMREAD_GRAYSCALE);
    }
}

