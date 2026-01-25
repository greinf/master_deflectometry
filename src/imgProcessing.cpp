#include "imgProcessing.hpp"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <math.h>
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
#include <numeric>
#include <algorithm>
#include <cmath>
#include "GrayCalibVector.hpp"
#include "utils.hpp"


auto normalizeAndDisplay = [](const cv::Mat& img) -> void {
    cv::Mat norm;
    cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Normalized", norm);
    cv::waitKey(0);
    cv::destroyWindow("Normalized");
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

std::vector<cv::Vec2d> ImageProcessing::harrisCornerDetection(
    const std::vector<cv::Mat>& img,
    const cv::Mat& mask,
    const cv::Size& window_sz,
    const double k1, 
    bool blur,
    bool gaussian_w)
{
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(!img.empty());
    CV_Assert(window_sz.area() > 0);
    CV_Assert(k1 > 0);
    if (img.size() == 1) return harrisCornerDetection(img[0], mask, window_sz, k1, blur, gaussian_w);

    cv::Mat mean_img = mean(img);
    return harrisCornerDetection(mean_img, mask, window_sz, k1, blur, gaussian_w);
}

std::vector<cv::Vec2d> ImageProcessing::refineCorner(
    const std::vector<cv::Mat>& img,
    const std::vector<cv::Vec2d>& corners,
    const cv::Size& sz,
    const double eps)
{
    CV_Assert(!img.empty());
    CV_Assert(!corners.empty());
    CV_Assert(std::all_of(img.begin(), img.end(),
        [&](const cv::Mat& pic) -> bool {
            return img.begin()->size() == pic.size();
        }));
    CV_Assert(std::all_of(corners.begin(), corners.end(),
        [&](const cv::Vec2d& vec) -> bool {
            // val[0] ist Y (Reihe)
            bool y_ok = (vec.val[0] > (sz.height / 2)) &&
                (vec.val[0] < (img[0].rows - sz.height / 2));

            // val[1] ist X (Spalte)
            bool x_ok = (vec.val[1] > (sz.width / 2)) &&
                (vec.val[1] < (img[0].cols - sz.width / 2));

            return x_ok && y_ok;
        }));
    CV_Assert(sz.area() < img[0].size().area());
    CV_Assert(eps > 0);

    cv::Mat mean_img = mean(img);

    return refineCorner(mean_img, corners, sz, eps);
}

std::vector<cv::Vec2d> ImageProcessing::refineCorner(
    const cv::Mat& img,
    const std::vector<cv::Vec2d>& corners,
    const cv::Size& sz,
    const double eps)
{
    CV_Assert(!img.empty());
    CV_Assert(!corners.empty());
    CV_Assert(std::all_of(corners.begin(), corners.end(),
        [&](const cv::Vec2d& vec) -> bool {
            // val[0] ist Y (Reihe)
            bool y_ok = (vec.val[0] > (sz.height / 2)) &&
                (vec.val[0] < (img.rows - sz.height / 2));

            // val[1] ist X (Spalte)
            bool x_ok = (vec.val[1] > (sz.width / 2)) &&
                (vec.val[1] < (img.cols - sz.width / 2));

            return x_ok && y_ok;
        }));
    CV_Assert(sz.area() < img.size().area());
    CV_Assert(eps > 0);
    CV_Assert(img.type() == CV_64F);
    CV_Assert(sz.height == sz.width);
    CV_Assert(sz.height % 2);


    std::vector<cv::Vec2d> corner_copy = corners;
    cv::Mat I_x, I_y;
    cv::Sobel(img, I_y, CV_64F, 1, 0);
    cv::Sobel(img, I_x, CV_64F, 0, 1);
    cv::Mat gaussian1D = cv::getGaussianKernel(sz.height, 0, CV_64F);
    cv::Mat gaussian = gaussian1D * gaussian1D.t();


    for (auto& corner : corner_copy) {
        
        for (int iteration = 0; iteration < 1000; ++iteration) {
            cv::Mat G(2, 2, CV_64F, cv::Scalar(0)); // names as in the openCV documentation
            cv::Mat b(2, 1, CV_64F, cv::Scalar(0));
            // Iterating steps
        // First Iteration are with interger coordinates - this allows to be on the grid. 
            const double img_start_row = corner[0] - static_cast<double>(sz.height / (int)2);
            const double img_start_cols = corner[1] - static_cast<double>(sz.width / (int)2);
            const double img_stop_row = corner[0] + static_cast<double>(sz.height / (int)2);
            const double img_stop_cols = corner[1] + static_cast<double>(sz.width / (int)2);

            const double* const gaus_ptr = gaussian.ptr<double>(0);
            int gaus_count{ 0 };
            for (double img_row = img_start_row; img_row <= img_stop_row; ++img_row) {
                for (double img_col = img_start_cols; img_col <= img_stop_cols; ++img_col) {
                    const double window_row = img_row - corner[0] - sz.height / 2;
                    const double window_col = img_col - corner[1] - sz.height / 2;

                    cv::Vec2d coord{ img_row, img_col };
                    // Holds the gradient for the given image Point
                    cv::Mat gradient = (cv::Mat_<double>(2, 1) <<
                        bilinearInterpolation(I_x, coord),
                        bilinearInterpolation(I_y, coord));
                    cv::Mat p = (cv::Mat_<double>(2,1) <<
                        img_col - corner[1], 
                        img_row - corner[0]);
                    /*cv::Mat diff_points = (cv::Mat_<double>(2, 1) <<
                        (corner[1] - img_col), (corner[0] - img_row));*/
                    //cv::Mat point = (cv::Mat_<double>(2, 1) << img_col, img_row);
                    if (gaus_count >= gaussian.size().area()) throw std::runtime_error("Gausindex out of bounds. ");
                    cv::Mat G_rc = gaus_ptr[gaus_count]*(gradient * gradient.t());
                    G += G_rc;
                    b += (G_rc * p);
                    gaus_count++;
                }
            }
            cv::Mat delta_sigma;
            cv::solve(G, b, delta_sigma, cv::DECOMP_LU); // Or DECOMP_CHOLESKY
            corner[1] -= delta_sigma.ptr<double>(0)[0];
            corner[0] -= delta_sigma.ptr<double>(0)[1];
            if (cv::norm(delta_sigma) < eps) break;
        }
    }
    return corner_copy;
}



// Use Bilinear Interpolation
// Expet the Vec2d to be (y,x)!
double ImageProcessing::bilinearInterpolation(
    const cv::Mat& img,
    const cv::Vec2d& coord)
{
    CV_Assert(img.type() == CV_64F);
    CV_Assert(img.size().area() > 2);
    CV_Assert(img.channels() == 1);
    CV_Assert(coord.val[0] >= 0 && coord.val[1] >= 0);
    CV_Assert(img.rows > coord.val[0] && img.cols > coord.val[1]);

    double floorX, ceilX, floorY, ceilY;
    cv::Mat coordMat(2, 2, CV_64F);
    cv::Mat vec1(1, 2, CV_64F), vec2(2, 1, CV_64F);
    double* first = coordMat.ptr<double>(0);
    double* vec1ptr = vec1.ptr<double>(0);
    double* vec2ptr = vec2.ptr<double>(0);

    floorY = std::floor(coord[0]);
    ceilY = std::ceil(coord[0]);
    
    floorX = std::floor(coord[1]);
    ceilX = std::ceil(coord[1]);

    bool sameY(floorY == ceilY);
    bool sameX(floorX == ceilX);
    
    // Validity check, Methode would fail for integer (nothing to interpolate)
    if(sameX) {
        if (sameY) return img.at<double>(static_cast<int>(coord[0]), 
            static_cast<int>(coord[1]));
        else {
            double val;
            double denom = ceilY - floorY;
            val = (ceilY - coord[0]) * img.at<double>(static_cast<int>(ceilY), static_cast<int>(coord[1])) / denom +
                (coord[0] - floorY) * img.at<double>(static_cast<int>(floorY), static_cast<int>(coord[1])) / denom;
            return val;
        }
    }

    if (sameY) {
        // same X was checked before
        double val;
        double denom = ceilX - floorX;
        val = (ceilX - coord[1]) * img.at<double>(static_cast<int>(coord[0]), static_cast<int>(ceilX)) / denom +
            (coord[1] - floorX) * img.at<double>(static_cast<int>(coord[0]), static_cast<int>(floorX)) / denom;
        return val;
    }

    const double* floor_row = img.ptr<double>(static_cast<int>(floorY));
    const double* ceil_row = img.ptr<double>(static_cast<int>(ceilY));

    first[0] = floor_row[static_cast<int>(floorX)];
    first[1] = ceil_row[static_cast<int>(floorX)];
    first[2] = floor_row[static_cast<int>(ceilX)];
    first[3] = ceil_row[static_cast<int>(ceilX)];

    double scalar = 1 / ((ceilX - floorX) * (ceilY - floorY));
    vec1ptr[0] = (ceilX - coord[1]);
    vec1ptr[1] = (coord[1] - floorX);

    vec2ptr[0] = (ceilY - coord[0]);
    vec2ptr[1] = (coord[0] - floorY);
    
    cv::Mat result = (vec1 * coordMat * vec2);
    if (result.size().area() == 1) return  scalar * result.at<double>(0, 0);
    else throw std::runtime_error("Bilinear Interpolation creates not reasonable values");
    return {};
}

std::vector<cv::Vec2d> ImageProcessing::findExtrema(
    const std::vector<cv::Vec2d>& corners,
    const std::vector<cv::Mat>& img,
    const cv::Size& sz)
{
    CV_Assert(!corners.empty() || !img.empty());
    CV_Assert(sz.area() > 0);
    CV_Assert(img[0].size().area() > sz.area());

    cv::Mat mean_img = mean(img);
    return findExtrema(corners, mean_img, sz);
}




// This was put on ice but can be resumed at later time. 
// 
// Here we try with a quaratik
// The goal is to approsimate a given window of a function through a polynom
// We try to find the extrema, hopefully the corner of the chessboard location
// q(u,v) = a*u^2 + b*u*v + c*v^2 + d*u + e*v + f
// q(u,v) = t^T * p
// t^T = (a,b,c,d,e,f)
// p^T = (u^2, u*v, v^2, u, v, 1)
// First goal approximate: 
// residum r = Sum (over (u,v) w(u,v) * (q(u,v) - I(u,v))^2
//
std::vector<cv::Vec2d> ImageProcessing::findExtrema(
    const std::vector<cv::Vec2d>& corners,
    const cv::Mat& img,
    const cv::Size& sz)
{
    CV_Assert(!corners.empty());
    CV_Assert(std::all_of(corners.begin(), corners.end(),
        [&img](const cv::Vec2d& vec) -> bool {
            return ((vec[0] < img.rows) && (vec[1] < img.cols));
        }));
    CV_Assert(sz.height == sz.width);

    // Build the gaussian window for weighting coords far from center differntly
    cv::Mat gaussian1D = cv::getGaussianKernel(sz.height, 0, CV_64F);
    cv::Mat gaussian = gaussian1D * gaussian1D.t();

    for (const auto& corner : corners) {
        
        //define the coordinates we are looking at
        double startRow = static_cast<double>(corner[0]) -
            static_cast<double>(sz.height)/2.0;
        double endRow = static_cast<double>(corner[0]) +
            static_cast<double>(sz.height) / 2.0;

        double startCols = static_cast<double>(corner[1]) -
            static_cast<double>(sz.width) / 2.0;
        double endCols = static_cast<double>(corner[1]) +
            static_cast<double>(sz.width) / 2.0;
        
        // Iterate to get better approximation
        cv::Mat coeffs(6, 1, CV_64F, cv::Scalar(1));
        cv::Mat coords(6, 1, CV_64F);

        //for (double row_img = startRow; row_img < endRow; ++row_img) {
        //    for (double cols_img = startCols; cols_img < endCols; ++cols_img) {
        //        // First we need to center the coordiantes. 
        //        double& co1 = 
        //        const double row_window = row_img - corner[0];
        //        const double cols_window = cols_img - corner[1];
        //        cv::Mat H(6, 6, CV_64F);
        //        H = coeffs * coeffs.t();
        //    }
        //}
        return {};

    }
    
}

// --- Build Jacobian to linearis the Problem
// x = x_0 + t_x + u * cos(aplpha) - v*sin(alpha)
// y = y_0 + t_x + u * sin(alpha) + v*cols(alpha)
// parameters theta = (tx,ty, alpha)
// res = sum(over (u,v)) w(u,v) * ((I(x(theta), x(theta)) - T(u,v)) = w(u,v) * I(x(theta), y(theta))-T
// linearisiere the problem (without we would not be able to otimize): 
// d(res)/d(theta)= w(u,v) * ( dI/dx * dx/d(theta) , dI/dy * dy/d(theta)) 
// res(theata + sigma) = r(theta) + dr/d(theata) * sigma
// this we can optimize
// min res = Sum(u,v) w(u,v)*(r+J*sigma)^2
// dres/dsigma = Sum(u,v) w(u,v)*2*(r+J*sigma)*J = 0 we optimize for derivative = 0
// dres/dsigma = 0 = sum(u,v) 2*w(u,v)*r*J^T = 2*w*J^T*J*sigma
// so we can solve this for sigma
// let H = w*J^T*J and let g= w*J^T*r => H*sgima = -g => sigma = H^-1*(-g) we iteratte over this solution 
// since the system is only real linear in a small area around the point
std::vector<cv::Vec2d> ImageProcessing::doTemplateMatching(
    std::vector<cv::Vec2d>& corners,
    const cv::Mat& img,
    const cv::Mat& templ)
{
    CV_Assert(img.type() == CV_64F);
    CV_Assert(templ.type() == CV_64F);
    CV_Assert((templ.rows % 2) && (templ.cols % 2));
    CV_Assert(!corners.empty());
    CV_Assert(img.size().area() > 100);
    CV_Assert(img.size().area() > templ.size().area());

    cv::Mat templ64F;
    if (templ.type() != CV_64F) templ.convertTo(templ64F, CV_64F);

    cv::Mat img_x, img_y;
    cv::Sobel(img, img_x, CV_64F, 1, 0);
    cv::Sobel(img, img_y, CV_64F, 0, 1);

    // --- Build the window Function Gaussian window
    cv::Mat gaussian1D = cv::getGaussianKernel(templ.size().height, 0, CV_64F);
    cv::Mat gaussian = gaussian1D * gaussian1D.t();
    
    // Just for testing just work with the frist found corner. 
    std::vector<cv::Vec2d> corners_copy{ corners[0] };
    
    for (auto& corner : corners_copy) {
        cv::Size size = templ.size();

        // define the starting points in the image plane
        // since templ.size() must be odd we do integer divition with 2 so the middle point is zero.
        const int startPointRow = static_cast<int>(corner[0])
            - size.height / 2;
        const int startPointCols = static_cast<int>(corner[1])
            - size.width / 2;
        const int stopPointRow = static_cast<int>(corner[0])
            + size.height / 2;
        const int stopPointCols = static_cast<int>(corner[1])
            + size.width / 2;

        
        cv::Vec3d sigma(0, 0, 0);
        
        // working variables as references to the sigma vector
        volatile double& tx = sigma[0];
        volatile double& ty = sigma[1];
        volatile double& alpha = sigma[2];

        //Build here the iteration loop!!!

        for (int iterations = 0; iterations < 100; ++iterations) {
            cv::Mat H(3, 3, CV_64F, cv::Scalar(0.0));
            cv::Mat g(3, 1, CV_64F, cv::Scalar(0.0));
            for (int temp_row = startPointRow; temp_row <= stopPointRow; ++temp_row) {
                for (int temp_cols = startPointCols; temp_cols <= stopPointCols; ++temp_cols) {

                    // shifted coordinates here coordinates U V are centered around the given Point we want to evaluate
                    const int v = temp_row - (static_cast<int>(corner[0]));
                    const int u = temp_cols - (static_cast<int>(corner[1]));

                    cv::Mat J = (cv::Mat_<double>(2, 3) <<
                        1.0, 0.0, -u * std::sin(alpha) - v * std::cos(alpha),
                        0.0, 1.0, u * std::cos(alpha) - v * std::sin(alpha));
                    // We need to calculate the Ix(u,v) Iy(u,v) and I(u,v). 
                    // Only Problem u,v are functions itself. We call it u_warp and v_warp
                    const double u_warp = corner[1] + tx + u * std::cos(alpha) - v * std::sin(alpha);
                    const double v_warp = corner[0] + ty + u * std::sin(alpha) + v * std::cos(alpha);

                    cv::Vec2d working_coord(v_warp, u_warp);

                    const double i_inter = bilinearInterpolation(img, working_coord);
                    const double i_x_inter = bilinearInterpolation(img_x, working_coord);
                    const double i_y_inter = bilinearInterpolation(img_y, working_coord);

                    cv::Mat IxIy = (cv::Mat_<double>(2, 1) << i_x_inter, i_y_inter);

                    cv::Mat jacobian = IxIy.t() * J; // this should have as a result (1x3)

                    // Template coordinate. Shifted that origin is like openCV in the left up corner
                    // Also used for the weighting with the gaussian window
                    const int v_templ = v + (size.height / 2);
                    const int u_templ = u + (size.width / 2);
                    const double r = i_inter - templ.at<double>(v_templ, u_templ);

                    // The formula we are minimizing for is the min sum(u,v) w(u,v)*(r+J*sigma)^2;
                    // Now create the building blocks H, g, sigma.
                    double window = gaussian.at<double>(v_templ, u_templ);

                    cv::Mat H_temp = window * (jacobian.t() * jacobian);
                    cv::Mat g_temp = window * r * jacobian.t();

                    H += H_temp;
                    g += g_temp;
                }
            }
            cv::Mat delta_sigma;
            cv::solve(H, -g, delta_sigma, cv::DECOMP_CHOLESKY); 

            const double* it_ptr = delta_sigma.ptr<double>(0);
            if (delta_sigma.size().area() == 3) {
                sigma.val[0] += it_ptr[0];
                sigma.val[1] += it_ptr[1];
                sigma.val[2] += it_ptr[2];
            }
            if (cv::norm(delta_sigma) < 1e-6) break;
            else {
                throw std::runtime_error("Return Vector of wrong dimension ");
            }
        }
        // corner is reference to the big verctor 
        corner = cv::Vec2d(corner[0] + sigma.val[1], corner[1] + sigma.val[0]);
    }
    return corners_copy;
}

cv::Mat ImageProcessing::getTemplateChess(
    const cv::Mat img,
    const cv::Size sz )
{
    CV_Assert(sz.height == sz.width);
    CV_Assert(!(img.rows % 2));
    CV_Assert(img.channels() == 1);
    CV_Assert(sz.area() < img.size().area());
    cv::Mat temp(sz, CV_64F);

    cv::Mat img64;
    if (img.type() != CV_64F) cv::normalize(img, img64, 0, 255, cv::NORM_MINMAX, CV_8U);
    else img64 = img;

    if (!(sz.height % 2)) {
        // -- define the starting point
        const int img_rowStart = (img.rows / 2) - (sz.height / 2);
        const int img_colsStart = (img.cols / 2) - (sz.width / 2);

        for (int row = 0; row < temp.rows; ++row) {
            for (int cols = 0; cols < temp.cols; ++cols) {
                temp.at<double>(row, cols) =
                    img64.at<double>(row + img_rowStart, cols + img_colsStart);
            }
        }
        /*cv::Mat show;
        cv::normalize(temp, show, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imshow("template", temp);
        cv::waitKey(0);*/

        return temp;
    }
    else {
        const double img_rowStart = 
            (img.rows / 2) - (static_cast<double>(sz.height) / 2.0);
        const double img_colsStart =
            (img.cols / 2) - (static_cast<double>(sz.height) / 2.0);

        for (int row = 0; row < temp.rows; ++row) {
            for (int cols = 0; cols < temp.cols; ++cols) {
                const cv::Vec2d coor{ row + img_rowStart, cols + img_colsStart };
                temp.at<double>(row, cols) =
                    bilinearInterpolation(img, coor);
            }
        }
        /*cv::Mat show;
        cv::normalize(temp, show, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imshow("template", temp);
        cv::imwrite("C:/Users/grein/Desktop/josepha.jpg", show);
        cv::waitKey(0);*/
        return temp;
    }

}

std::vector<cv::Vec2d> ImageProcessing::doTemplateMatching(
    std::vector<cv::Vec2d>& corner,
    const std::vector<cv::Mat>& img,
    const cv::Mat& templ)
{
    
    CV_Assert(!corner.empty());
    CV_Assert(!img.empty());
    CV_Assert((templ.cols % 2) && (templ.rows % 2));
    CV_Assert(std::all_of(img.begin(), img.end(),
        [](const cv::Mat& image) {
            return image.channels() == 1;
        }));
    CV_Assert(std::all_of(img.begin(), img.end(),
        [&](const cv::Mat& image) {
            return image.size() == img.begin()->size();
        }));
    CV_Assert(templ.size().area() < img[0].size().area());

    cv::Mat img_mean = mean(img);

    return doTemplateMatching(corner, img_mean, templ);
}




std::vector<cv::Vec2d> ImageProcessing::harrisCornerDetection(
    const cv::Mat& img,
    const cv::Mat& mask,
    const cv::Size& sz,
    const double k1,
    bool blur, 
    bool gaussian_window)
{
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(mask.size() == img.size());
    CV_Assert(img.size().area() > 0);
    CV_Assert(sz.area() > 0 && sz.area() < 100);
    CV_Assert(sz.height == sz.width);
    CV_Assert(sz.height % 2);
    CV_Assert(k1 > 0);
    CV_Assert(img.channels() == 1);


    cv::Mat working = img.clone();
    if (blur) cv::GaussianBlur(working, working, { 3,3 }, 0);

    cv::Mat working64F;
    if (working.type() != CV_64F) working.convertTo(working64F, CV_64F);
    else working64F = working;

    // --- Get d(img)/dx = img_x , d(img)/dy = img_y ---
    cv::Mat img_x, img_y;
    cv::Sobel(working64F, img_x, CV_64F, 1, 0, 5);
    cv::Sobel(working64F, img_y, CV_64F, 0, 1, 5);

    // --- Get helper for iterating the window
    std::vector<int> cols_helper(sz.width);
    std::iota(cols_helper.begin(), cols_helper.end(), -sz.width/(int)2);
    
    std::vector<std::pair<double, cv::Vec2d>> results;


    // --- Create a gaussian Window for weighting the differences
    cv::Mat window(sz, CV_64F, cv::Scalar(1));
    if (gaussian_window) {
        cv::Mat window1d = cv::getGaussianKernel(sz.height, 0);
        window = window1d * window1d.t();
    }

    // --- ignore the outer points ---
    int half = sz.width / 2;

    for (int row = half; row < img.rows - half; ++row) {

        std::vector<const double*> row_ptr_x;
        std::vector<const double*> row_ptr_y;
        std::vector<const uchar*> mask_ptr;
        row_ptr_x.reserve(sz.height);
        row_ptr_y.reserve(sz.height);
        mask_ptr.reserve(sz.height);

        for (int w_row = -half; w_row <= half; ++w_row) {
            row_ptr_x.push_back(img_x.ptr<double>(row + w_row));
            row_ptr_y.push_back(img_y.ptr<double>(row + w_row));
            mask_ptr.push_back(mask.ptr<uchar>(row + w_row));
        }

        // Checks if we work with valid picture Points 

        for (int cols = half; cols < img.cols - half; ++cols) {   

            // Checks for allowed mask values 
            if (std::any_of(mask_ptr.begin(), mask_ptr.end(),
                [&half, &cols](const uchar* mask_ptr) -> bool
                {
                    bool invalid = false;
                    for (int i = -half; i <= half; ++i) {
                        if (mask_ptr[cols + i] == 0) {
                            invalid = true;
                            break;
                        }
                    }
                    return invalid;
                })) continue;


            double Sxx = 0.0, Syy = 0.0, Sxy = 0.0;

            for (std::size_t r = 0; r < row_ptr_x.size(); ++r) {
                const double* px = row_ptr_x[r];
                const double* py = row_ptr_y[r];
                const double* wrow = window.ptr<double>((int)r);

                for (std::size_t cw = 0; cw < cols_helper.size(); ++cw) {
                    int dx = cols_helper[cw];

                    double ix = px[cols + dx];
                    double iy = py[cols + dx];
                    double w = wrow[cw];

                    Sxx += w * ix * ix;
                    Syy += w * iy * iy;
                    Sxy += w * ix * iy;
                }
            }

            double det = Sxx * Syy - Sxy * Sxy;
            double trace = Sxx + Syy;

            double R = det - k1 * trace * trace;   // <--- wichtig: Minus

            results.emplace_back(R, cv::Vec2d((double)row, (double)cols));
        }
    }

    if (results.begin() == results.end()) {
        std::cout << "Harris Corner detection did not find any valid values. \n";
        return {};
    }

    std::sort(results.begin(), results.end(),
        [](const std::pair<double, cv::Vec2d>& a,
            const std::pair<double, cv::Vec2d>& b) {
                return (a.first > b.first);
        });


    std::vector<cv::Vec2d> returnvec;
    double k_max = results.begin()->first;
    for (const std::pair<double, cv::Vec2d>& corner : results) {
        returnvec.push_back(corner.second);
        if (corner.first < k_max * 0.99) return returnvec;
    }

    std::cout << "We should never reach this path \n";
    return {};
}


std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> ImageProcessing::do_calibration_Points(
    const std::vector<cv::Mat>& unwrapped,
    const cv::Mat& mask,
    const cv::Vec2d& refPoint,
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
    CV_Assert((refPoint.val[0] > 0) && (refPoint.val[0] < unwrapped[0].rows));
    CV_Assert((refPoint.val[1] > 0) && (refPoint.val[1] < unwrapped[1].cols));


    std::vector<cv::Vec2d> imagePoints;
    std::vector<cv::Vec3d> objectPoints;
    std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> output;


    cv::Size sz = unwrapped[0].size();
    const double step_X = static_cast<double>(sz.width) / gridX;
    const double step_Y = static_cast<double>(sz.height) / gridY;
    //Debug image
    cv::Mat debug(mask.clone());

    cv::cvtColor(debug, debug, cv::COLOR_GRAY2BGR);

    // --- Get the intenisty values at the reference point ---
    const double phiRefX = bilinearInterpolation(unwrapped[0], refPoint);
    const double phiRefY = bilinearInterpolation(unwrapped[1], refPoint);
   
    // Sanity check if ref is invalid, you can't anchor the coordinate system
    CV_Assert(std::isfinite(phiRefX) && std::isfinite(phiRefY));
    // Checks if the referencePoint is marked as valid on the mask  
    cv::Mat mask64;
    mask.convertTo(mask64, CV_64F);
    CV_Assert(bilinearInterpolation(mask64, refPoint));

    // --- Debug ---
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    for (double rowd = 0; rowd < sz.height; rowd += step_Y) {
        int row = std::min(int(rowd), sz.height - 1);
        const double* x_ptr = unwrapped[0].ptr<double>(row);
        const double* y_ptr = unwrapped[1].ptr<double>(row);
        const uchar* mask_ptr = mask.ptr<uchar>(row);

        for (double colsd = 0; colsd < sz.width; colsd += step_X) {
            int cols = std::min(int(colsd), sz.width - 1);
            if (mask_ptr[cols] == 0) continue;

            // --- This can propbaly be discarded --- 
            if (row < 0 || cols < 0) throw std::runtime_error("Index out of bounds: index < 0");
            if (row >= sz.height || cols >= sz.width) throw std::runtime_error("Index out of bounds : index > 0");
           
            double phaseValX = x_ptr[cols];
            double phaseValY = y_ptr[cols];

            // --- do NOT discard negatives; just require finite ---
            if (!std::isfinite(phaseValX) || !std::isfinite(phaseValY)) continue;

            // --- anchor phases with respect to refPoint ---
            const double dphiX = phaseValX - phiRefX;
            const double dphiY = phaseValY - phiRefY;

            // --- convert phase difference -> relativ to origin in the display center! ---
            const double X_pixel_rel = (dphiX / CV_2PI) * wavelength;
            const double Y_pixel_rel = (dphiY / CV_2PI) * wavelength;

            // --- pixels -> mm ---
            const double X_mm_rel = X_pixel_rel * pixe_pitch_mm;
            const double Y_mm_rel = Y_pixel_rel * pixe_pitch_mm;

            // --- shift the object points that the origin is the upper left corner ---
            const double X_mm = X_mm_rel + (1920.0 / 2.0 * pixe_pitch_mm); //screenWidth_mm * 0.5;
            const double Y_mm = Y_mm_rel + (1080.0 / 2.0 * pixe_pitch_mm); //screenHeight_mm * 0.5;

            double X_pixel = (phaseValX / CV_2PI) * wavelength;
            double Y_pixel = (phaseValY / CV_2PI) * wavelength;
            
            minX = std::min(minX, X_mm);
            minY = std::min(minY, Y_mm);
            maxX = std::max(maxX, X_mm);
            maxY = std::max(maxY, Y_mm);

            imagePoints.emplace_back(cv::Vec2d(cols, row));
            objectPoints.emplace_back(X_mm, Y_mm, 0);
                
            /*std::cout << "ImagePoints: " << cv::Vec2d(cols, row) << '\n';
            std::cout << "ObjectPoints: " << cv::Vec3d(X_mm, Y_mm, 0) << '\n';*/

            cv::drawMarker(debug, cv::Point(cols, row), cv::Scalar(0, 255, 0), cv::MARKER_CROSS);
        }
    }

    std::cout << "Object Points: \n" <<
        "minVal: X: " << minX << " Y: " << minY << '\n' <<
        "maxVal: X: " << maxX << " Y: " << maxY << '\n';

    std::cout << "Image Points: \n" <<
        "maxVal: X: " << sz.width << " Y: " << sz.height << '\n';

    output.first = imagePoints;
    output.second = objectPoints;

    cv::normalize(debug, debug, 0, 255, cv::NORM_MINMAX, CV_8SC3);
    cv::imshow("debug", debug);
    cv::waitKey(0);
    cv::destroyWindow("debug");
    return output;
}

std::vector<cv::Vec2d> ImageProcessing::getRefinedCheckerboardCorner(
    const std::vector<cv::Mat> img,
    const cv::Size sz)
{
    CV_Assert(!img.empty());
    CV_Assert(sz.area() >= 1);
    CV_Assert(std::all_of(
        img.begin(), img.end(), [](const cv::Mat img) {
            return img.channels() < 2;
        }
    ));

    cv::Mat mean_img;
    if (img.size() > 1) {
        mean_img = mean(img);
    }
    else mean_img = img[0];

    std::vector<cv::Vec2d> corners;

    cv::Mat img8U;
    if (mean_img.type() != CV_8U) {
        cv::normalize(mean_img, img8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
    else img8U = mean_img;

    if (cv::findChessboardCorners(img8U, sz, corners)) {
        std::cout << "Found " << corners.size() << " chessboard corner \n";
        std::cout << "Refining ... \n";
        cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::Type::EPS, 100, 1.0E-5));
        return corners;
    }

    else {
        std::cout << "Corner detection failed \n";
        return {};
    }
}

std::array<double, (std::size_t)3> ImageProcessing::doWhiteBalance(
    const std::vector<cv::Mat>& vec,
    int roi_x,
    int roi_y,
    int roi_width,
    int roi_height) {
    //Be carefull, only valid for BayerRG setup

    CV_Assert(roi_x >= 0 && roi_y >= 0);
    CV_Assert(roi_width > 0 && roi_height > 0);
    CV_Assert(std::all_of(vec.begin(), vec.end(),
        [](const cv::Mat& mat) ->bool {
            return (!mat.empty() && (mat.type() == CV_8UC1));
        }));
    cv::Mat white_mean = mean(vec);
    cv::Size sz = white_mean.size();
    double g{}, b{}, r{};
    int g_n{}, b_n{}, r_n{};

    for (int row = roi_y; row < roi_y + roi_height; row += 2) {
        //ptr_G RGRGR ...
        //ptr_B GBGBG ...
        const double* ptr_R = white_mean.ptr<double>(row);
        const double* ptr_B = white_mean.ptr<double>(row + 1);
        for (int cols = roi_x; cols < roi_x + roi_width; ++cols) {
            if (cols % 2) {
                g += ptr_R[cols];
                b += ptr_B[cols];
                ++g_n;
                ++b_n;
            }
            else {
                r += ptr_R[cols];
                g += ptr_B[cols];
                ++r_n;
                ++g_n;
            }
        }
    }
    if (!g_n || !b_n || !r_n) {
        return { 1, 1, 1 };
    }

    const double meanR = r / r_n;
    const double meanB = b / b_n;
    const double meanG = g / g_n;

    return { meanG / meanR, 1.0, meanG / meanB };
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

std::vector<std::pair<double,double>> ImageProcessing::find_ParallelogramCorners(const cv::Mat& bin)
{
    CV_Assert(bin.type() == CV_8U);
    CV_Assert(!bin.empty());

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) throw std::runtime_error("no contour");

    // größte Kontur
    auto it = std::max_element(contours.begin(), contours.end(),
        [](auto& a, auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
    const auto& c = *it;

    // approx
    double peri = cv::arcLength(c, true);
    std::vector<cv::Point> poly;
    // epsilon je nach Bildqualität anpassen (typisch 0.5% bis 3% vom Umfang)
    cv::approxPolyDP(c, poly, 0.02 * peri, true);

    // Wenn nicht genau 4, versuche epsilon zu variieren oder fallback
    if (poly.size() != 4) {
        throw std::runtime_error("We need four Points try to variate the epsion in approxPolyDP");
    }

    // Debug
    cv::Mat drawing = bin.clone();
    /*cv::drawContours(drawing, contours, 0, cv::Scalar(125), 10);*/
    for (const auto& pt : poly) {
        cv::drawMarker(drawing, pt, cv::Scalar(125));
    }
    cv::imshow("Contour", drawing);
    cv::waitKey(0);

    std::vector<std::pair<double, double>> pts_;

    for (const auto& pt : poly) {
        pts_.push_back(std::pair<double, double>(pt.x, pt.y));
    }
    
    std::vector<std::pair<double, double>> pts{ orderTLTRBRBL_sumdiff(pts_) };

    return pts;
}


std::vector<std::pair<double, double>> ImageProcessing::orderTLTRBRBL_sumdiff(const std::vector<std::pair<double, double>>& p)
{
    auto sum = [](const std::pair<double, double>& q) { return q.first + q.second; };
    auto diff = [](const std::pair<double, double>& q) { return q.first - q.second; };

    std::vector<std::pair<double, double>> out(4);

    out[0] = *std::min_element(p.begin(), p.end(),
        [&](const std::pair<double, double>& a, const std::pair<double, double>& b) { return sum(a) < sum(b); }); // TL
    out[3] = *std::max_element(p.begin(), p.end(),
        [&](const std::pair<double, double>& a, const std::pair<double, double>& b) { return sum(a) < sum(b); }); // BR
    out[2] = *std::min_element(p.begin(), p.end(),
        [&](const std::pair<double, double>& a, const std::pair<double, double>& b) { return diff(a) < diff(b); }); // TR
    out[1] = *std::max_element(p.begin(), p.end(),
        [&](const std::pair<double, double>& a, const std::pair<double, double>& b) { return diff(a) < diff(b); }); // BL

    return out;
}

cv::Vec2d ImageProcessing::projectPoint_homography(
    const cv::Vec2d& img,
    const cv::Mat& homography)
{
    CV_Assert(homography.size() == cv::Size(3, 3));

    cv::Mat H(homography.size(), CV_64F);
    if (homography.type() != CV_64F) homography.convertTo(H, CV_64F);
    else H = homography;
    double x = H.at<double>(0, 0) * img[0] + H.at<double>(0, 1) * img[1] + H.at<double>(0, 2);
    double y = H.at<double>(1, 0) * img[0] + H.at<double>(1, 1) * img[1] + H.at<double>(1, 2);
    double w = H.at<double>(2, 0) * img[0] + H.at<double>(2, 1) * img[1] + H.at<double>(2, 2);

    return { static_cast<double>(x / w),
             static_cast<double>(y / w) };
}

// (srcPts, dstPts, method)
cv::Mat ImageProcessing::getHomographyMat(
    const std::vector<cv::Vec2d>& srcPts,
    const std::vector<cv::Vec2d>& dstPts,
    const int method)
{
    CV_Assert(srcPts.size() == dstPts.size());
    CV_Assert(srcPts.size() >= 4);

    std::vector<cv::Point2f> srcPtsFloat, dstPtsFloat;
    srcPtsFloat.reserve(srcPts.size());
    dstPtsFloat.reserve(dstPts.size());

    for (const auto& p : srcPts) srcPtsFloat.emplace_back((float)p[0], (float)p[1]);
    for (const auto& p : dstPts) dstPtsFloat.emplace_back((float)p[0], (float)p[1]);

    return cv::findHomography(srcPtsFloat, dstPtsFloat, method);
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
    std::vector<cv::Vec2d> img_pts;
    img_pts.reserve(caliPoints.first.size());
    for (const auto& v : caliPoints.first)
        img_pts.emplace_back(cv::Vec2d(v[0], v[1]));

    std::vector<cv::Vec2d> undist;

    cv::Size dist_sz = distCoeffs.size();
    const double* dist_ptr = distCoeffs.ptr<double>(0);
    // ---Check if distotion_coefficients are set. Mainly debugging ---
    if (dist_ptr[0] || dist_ptr[1] || dist_ptr[2] || dist_ptr[3]) {  // 
        std::cout << "In Reprojection: Use Undistort Pipeline \n";
        // --- DistortImagePoints --- 
        undist = distortImagePoints(img_pts, caliMatrix, distCoeffs);
        //cv::undistortImagePoints(img_pts, undist, caliMatrix, distCoeffs);
        //undist = img_pts;
        //undistortion The Wrong Direction here
        /*for (const auto& vec : img_pts) {
            undist.emplace_back(undistortImagePts(vec, caliMatrix, distCoeffs));
        }*/
    }

    else {
        undist = img_pts;
        std::cout << "Undistort Pipeline Skipped \n";
    }

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
        cv::Vec3d d_cam(undist[i][0], undist[i][1], 1.0);

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

std::vector<cv::Vec2d> ImageProcessing::distortImagePoints(
    const std::vector<cv::Vec2d>& imgPts,
    const cv::Mat& calimatrix,
    const cv::Mat& distcoeffs) 
{
    CV_Assert(!calimatrix.empty());
    CV_Assert(!distcoeffs.empty());
    CV_Assert(!imgPts.empty());

    cv::Range range(0, static_cast<int>(imgPts.size()));
    std::vector<cv::Vec2d> distorted(imgPts.size());

    cv::parallel_for_(range,
        [&](const cv::Range range) {
            for (int i = range.start; i < range.end; ++i) {
                distorted[i] = newtonSolverdistort(imgPts[i], calimatrix, distcoeffs);
            }
        });
    
    return distorted;
}

void ImageProcessing::getResponseCurve_perSection(
    const std::vector<cv::Mat>& gray_images,
    GrayCalibVector& borders,
    const int n_pics_perVal,
    const int n_steps)
{
    CV_Assert(!gray_images.empty());
    
    // if m_sections_img is empty no boundary were set
    CV_Assert(!borders.m_sections_img.empty());
    CV_Assert(n_pics_perVal >= 1);
    CV_Assert(n_steps >= 1);
    CV_Assert(gray_images.size() == 255 * n_pics_perVal / n_steps);

    std::vector<cv::Mat> mean_gray;
    
    if (n_pics_perVal > 1) {
        std::vector<cv::Mat>::const_iterator start = gray_images.begin();
        while (start != gray_images.end()) {
            if(std::distance(start, gray_images.end()) < n_pics_perVal) {
                throw std::runtime_error("Iterator out of bounds, getResponseCurve_perSection \n");
            }
            std::vector<cv::Mat>::const_iterator end = std::next(start, n_pics_perVal);
            mean_gray.emplace_back(mean(std::vector<cv::Mat>(start, end)));
            start = end;
        }
    }
    else {
        for (const auto& img : gray_images) {
            cv::Mat img64;
            img.convertTo(img64, CV_64F);
            mean_gray.push_back(img64);
        }
    }

    CV_Assert(!mean_gray.empty());

    // Mask from first and last gray image
    std::vector<cv::Mat> getMask{ mean_gray[0], mean_gray[mean_gray.size() - 1] };
    cv::Mat mask = grayCalibMask(getMask);

    normalizeAndDisplay(mask);

    std::vector<std::pair<double, double>> mask_corners = find_ParallelogramCorners(mask);
    if (mask_corners.size() != 4) {
        std::cout << "There must be 4 Corners for the Parallelogram \n";
        throw std::runtime_error("There must be 4 Corners for the Parallelogram \n");
    }

    std::vector<std::pair<int, int>> img_borderPts = borders.getFullBorders_img();

    cv::Mat homography_Mat = getHomographyMat(img_borderPts, mask_corners);

    cv::parallel_for_(cv::Range(0, static_cast<int>(borders.m_sections_img.size())),
        [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                Gray_section_data& data = borders.m_sections_img[i];
                evaluateSection_gray(mean_gray, data, homography_Mat, mask);
            }
        }
    );

    /*for (std::size_t i = 0; i < borders.m_sections_img.size(); ++i) {
        Gray_section_data& data = borders.m_sections_img[i];
        
        evaluateSection_gray(mean_gray, data, homography_Mat, mask);
        std::cout << "Finished Sections " << i << "/" << borders.m_sections_img.size() << std::endl;
    }*/
}

void ImageProcessing::evaluateSection_gray(
    const std::vector<cv::Mat>& images,
    Gray_section_data& data_out,
    const cv::Mat& homography,
    const cv::Mat& mask)
{
    CV_Assert(!images.empty());
    CV_Assert(data_out.gray_val.empty());
    CV_Assert(!homography.empty());
    CV_Assert(!mask.empty());
    CV_Assert(images[0].type() == CV_64F);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(std::all_of(images.begin(), images.end(), [&](const cv::Mat& img) {
        return img.size() == mask.size();
        }));

    // The datapoints are stored as (x,y) 
    int start_col = data_out.roi_img.left_up_corner.first;
    int end_col = data_out.roi_img.right_up_corner.first;
    int start_row = data_out.roi_img.left_up_corner.second;
    int end_row = data_out.roi_img.left_down_corner.second;

    std::vector<double> means;
    std::vector<double> stdevs;
    means.reserve(images.size());
    stdevs.reserve(images.size());

    // bilinear Interpolation expects CV_64F images. 
    cv::Mat mask64F;
    mask.convertTo(mask64F, CV_64F);

    for (const auto& img : images) {
        double mean = 0.0;
        double M2 = 0.0;
        int n = 0;

        for (int col = start_col; col < end_col; ++col) {
            for (int row = start_row; row < end_row; ++row) {

                cv::Vec2d p_img(
                    static_cast<double>(col),
                    static_cast<double>(row));

                cv::Vec2d p_proj = projectPoint_homography(p_img, homography);

                // bilinear expects (y, x)
                cv::Vec2d p_yx(p_proj[1], p_proj[0]);

                if (p_yx[0] >= img.rows || p_yx[1] >= img.cols || p_yx[0]< 0 || p_yx[1] < 0) {
                    continue; 
                }

                double val = bilinearInterpolation(img, p_yx);

                double val_mask = bilinearInterpolation(mask64F, p_yx);
                
                if (val_mask < 0.5) {
                    //checkHomogrpahy(img, homography, p_img);
                    continue;
                }
                ++n;
                double delta = val - mean;
                mean += delta / n;
                double delta2 = val - mean;
                M2 += delta * delta2;
            }
        }

        double variance = (n > 1) ? (M2 / (n - 1)) : 0.0;
        double stdev = std::sqrt(variance);

        means.push_back(mean);
        stdevs.push_back(stdev);
    }
    data_out.gray_val = means;
    data_out.s_deviation = stdevs;
}

void ImageProcessing::checkHomogrpahy(
    const cv::Mat& img,
    const cv::Mat& homography,
    const cv::Vec2d& coord) 
{
    CV_Assert(!img.empty() || !homography.empty());
    CV_Assert(img.type() == CV_64F);

    cv::Vec2d projected = projectPoint_homography(coord, homography);
    cv::Mat dummy_display(img.size(), CV_8U, cv::Scalar(0));
    cv::Mat camera64 = img.clone();
    cv::Mat camera;
    cv::normalize(camera64, camera, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::drawMarker(dummy_display, cv::Point(static_cast<int>(coord[0]),
        static_cast<int>(coord[1])), cv::Scalar(255), 0, 50);
    cv::drawMarker(camera, cv::Point(static_cast<int>(projected[0]), 
        static_cast<int>(projected[1])), cv::Scalar(255), 0, 50);
    cv::imshow("Dispaly", dummy_display);
    cv::imshow("Projected", camera);
    cv::waitKey(0);
   
}

cv::Mat ImageProcessing::createCirculeBinaryMask(
    const double radius)
{
    CV_Assert(radius > 0.0);
    int r = static_cast<int>(std::ceil(radius));
    int k = 2 * r + 1;
    cv::Mat se(k, k, CV_8U, cv::Scalar(0));

    for (int y = 0; y < k; ++y) {
        for (int x = 0; x < k; ++x) {
            double dx = x - r; // Shift origin in the middle 
            double dy = y - r;
            if (dx * dx + dy * dy <= r)
                se.at<uint8_t>(y, x) = 1; // oder 255
        }
    }
    return se;
}


cv::Mat ImageProcessing::getHomographyMat(
    const RoiBorders<int> src,
    const RoiBorders<double> dst,
    const int method ) 
{
    // source Imagepoints
    std::vector<std::pair<int, int>> source{};
    source.push_back(src.left_up_corner);
    source.push_back(src.right_up_corner);
    source.push_back(src.left_down_corner);
    source.push_back(src.right_down_corner);

    // destination Objectpoints

    std::vector<std::pair<double, double>> destination{};
    destination.push_back(dst.left_up_corner);
    destination.push_back(dst.right_up_corner);
    destination.push_back(dst.left_down_corner);
    destination.push_back(dst.right_down_corner);

    return getHomographyMat(source, destination, method);
}



cv::Mat ImageProcessing::getHomographyMat(
    const std::vector<std::pair<int, int>>& src,
    const std::vector<std::pair<double, double>>& dst,
    const int method
)
{
    CV_Assert(src.size() == dst.size());
    CV_Assert(!src.empty() && !dst.empty());

    std::vector<cv::Vec2d> src_n;
    std::vector<cv::Vec2d> dst_n;
    std::cout << "Image Points for Homography: \n" << "Must ge Left to right, up -> down\n";
    for (const auto& s_pts : src) {
        src_n.emplace_back(cv::Vec2d(s_pts.first, s_pts.second));
        std::cout << "x: " << s_pts.first << "y: " << s_pts.second << '\n';
    }
    std::cout << "Object Point for Homography: \n" << "Must be left to right and up to down \n";
    for (const auto& d_pts : dst) {
        dst_n.emplace_back(cv::Vec2d(d_pts.first, d_pts.second));
        std::cout << "x: " << d_pts.first << "y: " << d_pts.second << '\n';
    }

    return getHomographyMat(src_n, dst_n);
}

cv::Mat ImageProcessing::simulate_luminance(
    const cv::Mat& img,
    const double distance,
    const double pixel_pitch,                //PixelPitch  FH 0.277    BMZ: ,
    const int display_pixel_x,
    const int display_pixel_y,
    const double shift_x,
    const double shift_y,
    const double angle_x,
    const double angle_y,
    const double(*funct_ptr)(double))
{
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_64F || img.type() == CV_8U);
    CV_Assert(distance > 0 && pixel_pitch > 0);
    CV_Assert(display_pixel_x > 0 && display_pixel_y > 0);
    CV_Assert(shift_x < distance && shift_y < distance);
    
    cv::Mat angle_img = angle_camera_toScreenNormal(
        img.size(),
        pixel_pitch,
        shift_x,
        shift_y,
        angle_x,
        angle_y,
        distance
    );
    
    for (int row = 0; row < angle_img.rows; ++row) {
        double* img_ptr = angle_img.ptr<double>(row);
        for (int col = 0; col < angle_img.cols; ++col) {
            if (funct_ptr == nullptr) img_ptr[col] = std::cos(img_ptr[col]); // lambert scaling
            else img_ptr[col] = funct_ptr(img_ptr[col]);
        }
    }
    cv::Mat output;
    cv::Mat img64;
    if (img.type() != CV_64F) img.convertTo(img64, CV_64F);
    else img64 = img;

    cv::multiply(angle_img, img64, output);

    return output;
    
}

cv::Mat ImageProcessing::calcCoordinateImage(
    const cv::Size& sz,
    const double pixel_pitch_disp,
    const double shift_x,
    const double shift_y)
{
    CV_Assert(sz.area() > 0 && pixel_pitch_disp > 0);
    CV_Assert(sz.height % 2 == 0 && sz.width % 2 == 0);

    cv::Mat_<cv::Vec3d> coordinates(sz, CV_64FC3);

    cv::parallel_for_(cv::Range(0, sz.height),
        [&](const cv::Range& range)
        {
            for (int start = range.start; start < range.end; ++start) {
                double world_y = (static_cast<double>(start - sz.height / 2) - shift_y) * pixel_pitch_disp;
                cv::Vec3d* row = coordinates.ptr<cv::Vec3d>(start);
                for (int col = 0; col < sz.width; ++col) {
                    double world_x = (static_cast<double>(col - sz.width / 2) - shift_x) * pixel_pitch_disp;
                    row[col] = cv::Vec3d(world_x, world_y, 0);
                }
            }
        });
    return coordinates;
}

cv::Mat ImageProcessing::rotateCoordinatedGrid(
    const cv::Mat& img,
    const cv::Vec3d& rotVector)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 3);
   
    if (rotVector[0] == 0 && rotVector[1] == 0 && rotVector[2] == 0) return img;

    cv::Matx33d Rx(
        1, 0, 0,
        0, std::cos(rotVector[0]), -sin(rotVector[0]),
        0, std::sin(rotVector[0]), std::cos(rotVector[0])
    );

    cv::Matx33d Ry(
        std::cos(rotVector[1]), 0, std::sin(rotVector[1]),
        0, 1, 0,
        -std::sin(rotVector[1]), 0, std::cos(rotVector[1])
    );

    // choose order: usually Ry * Rx (but depends on your convention)
    cv::Matx33d R = Ry * Rx;

    cv::Mat_<cv::Vec3d> output(img.rows, img.cols);

    cv::parallel_for_(cv::Range(0, img.rows),
        [&](const cv::Range& range) {
            for (int start = range.start; start < range.end; ++start) {
                const cv::Vec3d* ptr = img.ptr<cv::Vec3d>(start);
                cv::Vec3d* output_ptr = output.ptr<cv::Vec3d>(start);
                for (int cols = 0; cols < img.cols; ++cols) {
                    output_ptr[cols] = R * ptr[cols];
                }

            }
        });


    return output;
}



cv::Mat ImageProcessing::angle_camera_toScreenNormal(
    const cv::Size& sz,
    const double pixel_pitch_disp,
    const double shift_x,
    const double shift_y,
    const double angle_x,
    const double angle_y,
    const double distance)
{
    CV_Assert(sz.area() > 0 && pixel_pitch_disp > 0);
    CV_Assert(sz.height % 2 == 0 && sz.width % 2 == 0);
    
    cv::Mat_<cv::Vec3d> coordinates = calcCoordinateImage(
        sz,
        pixel_pitch_disp,
        0,  // set this alwasy at zero 
        0);


    cv::Mat_<cv::Vec3d> rotated = rotateCoordinatedGrid(
        coordinates,
        cv::Vec3d(angle_x, angle_y, 0)
    );

    // now Calculate the Anlge between Screen and Hitting Ray
    // |a| * |b| * sin(alpha) = a * b 
    // a is Vector on the image center to image point.
    // b is Vector from camera to image Point ->  (shift_x, shift_y,0) + (pixel_x, pixel_y, pixel_z (in case of rotation)) - distance

    cv::Mat output(sz.height, sz.width, CV_64F);

    cv::Vec3d X_axis = rotated.ptr<cv::Vec3d>(rotated.rows / 2)[rotated.cols-1];
    cv::Vec3d Y_axis = rotated.ptr<cv::Vec3d>(rotated.rows - 1)[rotated.cols / 2];

    cv::Vec3d display_normal = X_axis.cross(Y_axis);

    cv::parallel_for_(cv::Range(0, rotated.rows),
        [&](const cv::Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                cv::Vec3d* ptr = rotated.ptr<cv::Vec3d>(row);
                double* out_ptr = output.ptr<double>(row);
                for (int col = 0; col < rotated.cols; ++col) {
                    
                    cv::Vec3d b = ptr[col] - cv::Vec3d(shift_x, shift_y, -distance);
                    double denom = cv::norm(display_normal) * cv::norm(b);
                    if (denom == 0.0) { out_ptr[col] = 0; continue; }
                    double c = (display_normal.ddot(b)) / denom;
                    double angle = std::acos(c);
                    out_ptr[col] = angle;
                }
            }
        });

    /*cv::Mat img8u;
    cv::normalize(output, img8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("angel", img8u);
    cv::waitKey(0);*/


    return output;
}


cv::Mat ImageProcessing::grayCalibMask(const std::vector<cv::Mat>& img) {
    CV_Assert(!img.empty());
    CV_Assert(img.size() == 2);

    std::vector<cv::Mat> img64;
    if (img[0].type() != CV_64F) {
        for (const auto& image : img) {
            cv::Mat mat64F;
            image.convertTo(mat64F, CV_64F);
            img64.push_back(mat64F);
        }
    }
    else img64 = img;

    cv::Mat mask = img64[1] - img64[0];
    cv::Mat mask8u;
    cv::normalize(mask, mask8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    cv::threshold(mask8u, mask8u, 255 * 0.4, 255, CV_8U);
    return mask8u;
}


// Create Matrix A (x,y,1) for z(x,y) = a*x+b*x+c
// so the best fit shoud be 0 = z(x,y) - (a*x+b*y+c) for all points
// Therefor we try to solve for a,b,c
// We build a A as a Nx3 matrix (N points in ROI)
// Z(x,y) as the values in the real image 
cv::Mat ImageProcessing::fitSurface(
    const cv::Mat& img,
    const cv::Mat& mask)
{
    CV_Assert(img.size() == mask.size());
    CV_Assert(img.type() == CV_64F);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(img.channels() == 1 && mask.channels() == 1);

    int n_points = cv::countNonZero(mask);
    CV_Assert(n_points > 0 && n_points <= img.size().area());
    cv::Mat A, b, y, out;
    out = cv::Mat::zeros(img.size(), CV_64F);

    A.create(n_points, 3, CV_64F);
    b.create(3, 1, CV_64F);
    y.create(n_points, 1, CV_64F);

    cv::Moments mask_moments = cv::moments(mask, true);
    int origin_x = static_cast<int>(
        std::round(mask_moments.m10 / mask_moments.m00));
    int origin_y = static_cast<int>(
        std::round(mask_moments.m01 / mask_moments.m00));

    
    int point_counter{ 0 };
    for (int row = 0; row < img.rows; ++row) {
        const uchar* mask_ptr = mask.ptr<uchar>(row);
        const double* img_ptr = img.ptr<double>(row);
       
        for (int cols = 0; cols < img.cols; ++cols) {
            if (!mask_ptr[cols]) continue;
            double* a_ptr = A.ptr<double>(point_counter);
            double* y_ptr = y.ptr<double>(point_counter);
            int A_row[3] = { 
                row - origin_y,
                cols - origin_x,
                1 
            };
            for (int i = 0; i < 3; ++i) {
                a_ptr[i] = static_cast<double>(A_row[i]);
            }
            *y_ptr = img_ptr[cols];
            ++point_counter;
        }
    }

    // A(y,x,1)
    // Solve least squares: A * b = y
    cv::solve(A, y, b, cv::DECOMP_QR);

    const double a = b.at<double>(0);
    const double c = b.at<double>(1);
    const double d = b.at<double>(2);

    for (int row = 0; row < img.rows; ++row) {
        double* out_ptr = out.ptr<double>(row);
        const uchar* mask_ptr = mask.ptr<uchar>(row);
        for (int col = 0; col < img.cols; ++col) {
            if (!mask_ptr[col]) continue;
            out_ptr[col] = a * (row - origin_y) + c * (col - origin_x) + d;
        }
    }

    //normalizeAndDisplay(out);
    return out;
}

cv::Mat ImageProcessing::calcDistortionError(const cv::Mat& img) {
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_64FC2);

    cv::Mat error_img(img.size(), CV_64FC2);
    for (int row = 0; row < img.rows; ++row) {
        const double* pic = img.ptr<double>(row);
        double* error_pic = error_img.ptr<double>(row);
        for (int cols = 0; cols < img.cols; ++cols) {
            error_pic[cols * 2] = pic[cols * 2] - static_cast<double>(cols);
            error_pic[cols * 2 + 1] = pic[cols * 2 + 1] - static_cast<double>(row);
        }
    }

    return error_img;
}

cv::Vec2d ImageProcessing::undistortImagePts(
    const cv::Vec2d img_pt,
    const cv::Mat& K,
    const cv::Mat& dist_coeffs
)
{
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double k1 = dist_coeffs.at<double>(0, 0);
    double k2 = dist_coeffs.at<double>(1, 0);
    double p1 = dist_coeffs.at<double>(2, 0);
    double p2 = dist_coeffs.at<double>(3, 0);
    double k3 = dist_coeffs.cols >= 5 ? dist_coeffs.at<double>(4, 0) : 0.0;

    
    // --- Normalize in Camera Coordinates
    double x_u = (img_pt[0] - cx) / fx;
    double y_u = (img_pt[1] - cy) / fy;

    double r2 = x_u * x_u + y_u * y_u;
    double r4 = r2 * r2;
    double r6 = r4 * r2;

    double radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;

    // see lateral + radial distortion 
    double x_d = x_u * radial + 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u);
    double y_d = y_u * radial + p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u;

    double x_d1 = fx * x_d + cx;
    double y_d1 = fy * y_d + cy;

    return cv::Vec2d{ x_d1,y_d1 };
}
    


cv::Mat ImageProcessing::undistortImage(const cv::Mat& img,
    const cv::Mat& K,
    const cv::Mat& distCoeffs)
{
    int W = img.cols;
    int H = img.rows;

    cv::Mat map_x(H, W, CV_32F);
    cv::Mat map_y(H, W, CV_32F);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cv::Vec2d distorted = undistortImagePts(cv::Vec2d(x, y), K, distCoeffs);
            map_x.at<float>(y, x) = static_cast<float>(distorted[0]);
            map_y.at<float>(y, x) = static_cast<float>(distorted[1]);
        }
    }

    cv::Mat img32;
    img.convertTo(img32, CV_32F);

    cv::Mat distorted;
    cv::remap(img32, distorted, map_x, map_y, cv::INTER_CUBIC, cv::BORDER_CONSTANT);
    cv::Mat distorted64;
    distorted.convertTo(distorted64, CV_64F);

    return distorted64;
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
    CV_Assert(dist_coeffs.isContinuous());

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
    for (int iter = 0; iter < 1000; ++iter) {
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
            //std::cout << "Reached termination cirterium \n";
            break; // numerisch instabil -> abbrechen
        }

        // Newton step Δx, Δy (solve J * Δ = R)
        const double dx = (d * R_x - b * R_y) / det;
        const double dy = (-c * R_x + a * R_y) / det;

        x_u -= dx;
        y_u -= dy;

        if (dx * dx + dy * dy < 1e-18) {
            //std::cout << "Reached termination cirterium \n";
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
    const cv::Mat& mask) 
{
    CV_Assert(!wrapped.empty());
    //CV_Assert(wrapped[0].type() == mask.type());
    CV_Assert(wrapped[0].size() == mask.size());


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
    double threshold,
    bool dilate) 
{
    cv::Mat maskbin, sum_contrast_n;
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
    //normalizeAndDisplay(maskbin);
    // Create Structuring Element for opening&closing
    cv::Mat strucutre = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    cv::morphologyEx(maskbin, maskbin, cv::MORPH_OPEN, strucutre, cv::Point2d(-1, -1), 2);
    cv::morphologyEx(maskbin, maskbin, cv::MORPH_CLOSE, strucutre, cv::Point2d(-1, -1), 2);

    // Shrink the allowed values since we have a reference Point that is in the middle of the image.
    if(dilate) cv::morphologyEx(maskbin, maskbin, cv::MORPH_ERODE, strucutre, cv::Point2d(-1, -1), 50);
   
    normalizeAndDisplay(maskbin);

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
    cv::Mat structur = cv::getStructuringElement(cv::MORPH_RECT, { 5,5 });
    cv::erode(mask, mask, structur, cv::Point(-1, -1), 10);
    cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX, CV_8U);
    minmaxloc mask_info{ get_minmaxloc(mask) };
    cv::threshold(mask, mask, mask_info.maxval * 0.5, 255, cv::THRESH_BINARY);
    normalizeAndDisplay(mask);

    std::vector<cv::Scalar_<double>> mean_values;

    for (std::size_t i = 0; i < mean_vec.size(); ++i) {
        double GTgray_val = static_cast<double>(i * stepwidth);
        double meassure_gray_val = cv::norm(cv::mean(mean_vec[i], mask));
        LUT.push_back(std::pair<double, double>(GTgray_val, meassure_gray_val));
    }

    return LUT;
}


std::vector<cv::Mat> ImageProcessing::bayerToGray(
    const std::vector<cv::Mat>& img)
{
    CV_Assert(std::all_of(img.begin(), img.end(), [](const cv::Mat& img)
        {
            return img.type() == CV_8U;
        }));

    std::vector<cv::Mat> gray(img.size());

    for (std::size_t i = 0; i < img.size(); ++i) {
		cv::cvtColor(img[i], gray[i], cv::COLOR_BayerRG2GRAY);
    }
    return gray;
}

auto normalize = [](const cv::Mat& pic) -> cv::Mat_<uchar> {
    if (pic.type() == CV_8U) return pic;
    cv::Mat_<uchar> dest;
    cv::normalize(pic, dest, 0, 255, cv::NORM_MINMAX, CV_8U);
    return dest;
    };


std::vector<double> ImageProcessing::extract_Row(
    int row,
    const cv::Mat& img,
    const cv::Mat& mask) 
{
    const cv::Mat& repro = m_unwrapped_phase[0];
    std::vector<double> row_unwrap;
    row_unwrap.reserve(repro.cols);
    for (int i = 0; i < repro.cols; ++i) {
        if (!m_mask.at<uchar>(row, i)) continue;
        row_unwrap.push_back(repro.at<double>(row, i));
    }
    return row_unwrap;
}


std::vector<double> ImageProcessing::extract_Column(
    int x,
    const cv::Mat& img,
    const cv::Mat& mask) 
{
    const cv::Mat& unwrap = m_unwrapped_phase[1];
    std::vector<double> col;
    col.reserve(unwrap.rows);

    for (int i = 0; i < unwrap.rows; ++i) {
        if (!m_mask.at<uchar>(i, x)) continue;
        col.push_back(unwrap.at<double>(i, x));
    }
    return col;
}

ImageProcessing::~ImageProcessing() {}

std::vector<cv::Mat> ImageProcessing::unwrapped_phase(
    const std::vector<cv::Mat>& wrappedPhase,
    const cv::Mat& mask) 
{
    CV_Assert(wrappedPhase.size() == 2);
    CV_Assert(wrappedPhase[0].size() == mask.size());
    CV_Assert(wrappedPhase[0].type() == CV_32F || wrappedPhase[0].type() == CV_64F);

    cv::Size sz = wrappedPhase[0].size();

    cv::Mat wX, wY, cX, cY;
    wrappedPhase[0].convertTo(wX, CV_32F);
    wrappedPhase[1].convertTo(wY, CV_32F);


    //normalizeAndDisplay(wX);
    std::vector<cv::Mat> unwrapped(2);
    unwrapped[0] = cv::Mat(sz, CV_64F);
    unwrapped[1] = cv::Mat(sz, CV_64F);
    cv::Mat unwrapX;
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params params;
    params.width = sz.width;
    params.height = sz.height;
    params.histThresh = static_cast<float>(CV_PI / 10.0);
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
    paramsY.histThresh = static_cast<float>(CV_PI / 10.0);
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
    CV_Assert(std::all_of(vec.begin(), vec.end(),
        [](const cv::Mat& mat) -> bool 
        { return mat.channels() == 1; }
    ));

    if (vec.size() == 1) return vec[0];

    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            throw std::runtime_error ("All pictures must be same kind and type. \n");
        }
    }
    cv::Mat acc;

    if(vec[0].type() != CV_64F) vec[0].convertTo(acc, CV_64F);
    
    for (size_t i = 0; i < vec.size(); ++i) {
        cv::Mat temp;
        vec[i].convertTo(temp, CV_64F);
        acc += temp;   // pixelweise Addition
    }

    acc /= static_cast<double>(vec.size());  // pixelweise Division

    return acc;
}


// Depracted code

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

//void ImageProcessing::saveImages_png(const std::string& path_s) {
//    std::filesystem::path path{ path_s };
//    if (!std::filesystem::exists(path)) {
//        if (!std::filesystem::create_directory(path)) std::cerr << "Creating the directory failed. \n";
//        std::cout << "Created Directory for the .png files \n";
//    }
//    else {
//        std::cout << "Directory for .png files already exists \n" << "Possible override. Do you want to proceed? \n [Y/N]";
//        char c;
//        bool cond{ true };
//        do {
//            std::cin >> c;
//            if (!std::cin.fail()) {
//                std::cin.clear();
//                std::cin.ignore(100, '\n');
//            }
//            if (std::toupper(c) == 'N') {
//                return;
//            }
//            if (std::toupper(c) == 'Y') {
//                cond = false;
//            }
//        } while (cond);
//    }
//    //Create save structure
//    std::array<std::string, 7> strings{ "Full_frames", "Mean_frames", "Base Intensity", "Contrast",
//        "WrappedPhase", "unwrappedPhase", "reprojectionError" };
//
//    for (std::size_t i = 0; i < strings.size(); ++i)
//    {
//        if (!std::filesystem::create_directory(path / strings[i])) {
//            std::cout << "Creating directory " << (path / strings[i]).string() << " failed \n";
//        }
//        switch (i) {
//        case(0): { save_png(path / strings[i], m_frames); break; }
//        case(1): { save_png(path / strings[i], m_raw_phase); break; }
//        case(2): { save_png(path / strings[i], m_baseIntensity); break; }
//        case(3): { save_png(path / strings[i], m_contrast); break; }
//        case(4): { save_png(path / strings[i], m_wrapped_phase); break; }
//        case(5): { save_png(path / strings[i], m_unwrapped_phase); break; }
//        case(6): { save_png(path / strings[i], m_reprojection_error_img); break; }
//
//        }
//    }
//}



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


//void ImageProcessing::goldsteinUnwrap() {
//    cv::Mat unwrapped1;
//    cv::Mat unwrapped2;
//    for (auto& m : m_unwrapped_phase) {
//        cv::normalize(m, m, 0, 1, cv::NORM_MINMAX, CV_32F);
//    }
//    cv::Mat mask1 = (m_contrast[0] > 0.2f); // for example: keep only valid contrast regions
//    mask1.convertTo(mask1, CV_8U);           // ensure binary mask
//    cv::Mat mask2 = (m_contrast[1] > 0.2f);
//    mask2.convertTo(mask2, CV_8U);
//    goldsteinUnwrapCV(m_wrapped_phase[0], unwrapped1, mask1);
//    goldsteinUnwrapCV(m_wrapped_phase[1], unwrapped2, mask2);
//
//    cv::normalize(unwrapped1, m_unwrapped_phase[0], 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::normalize(unwrapped2, m_unwrapped_phase[1], 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("Goldstein Unwrapped Phase 1", m_unwrapped_phase.at(0));
//    cv::imshow("Goldstein Unwrapped Phase 2", m_unwrapped_phase.at(1));
//    cv::waitKey(0);
//}
