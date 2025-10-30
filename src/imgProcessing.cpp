#include "imgProcessing.hpp"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <math.h>
#include <imageHandler.hpp>
#include <opencv2/phase_unwrapping/histogramphaseunwrapping.hpp>
#include "GoldsteinWrapper.hpp"
#include <filesystem>
#include "cvDepthTraits.hpp"


//Debugging function to calculate gradient strength
auto grad_strength = [](const cv::Mat& f) -> std::pair<cv::Mat, cv::Mat> {
    cv::Mat fx, fy;
    std::pair<cv::Mat, cv::Mat> pair;
    cv::Sobel(f, fx, CV_32F, 1, 0, 3); // d/dx
    cv::Sobel(f, fy, CV_32F, 0, 1, 3); // d/dy
    pair.first = fx;
    pair.second = fy;
    
    return pair;
    /*
    return std::pair<double, double>(
        cv::mean(cv::abs(fx))[0],
        cv::mean(cv::abs(fy))[0]
        */
    };


ImageProcessing::minmaxloc ImageProcessing::get_minmaxloc(cv::Mat& mat) const {
    minmaxloc helper;
    cv::minMaxLoc(mat, &helper.minval, &helper.maxval, &helper.minloc, &helper.maxloc);
    return helper;
}

ImageProcessing::ImageProcessing() {
    ++instance_counter;
    if (instance_counter > 1) throw std::runtime_error("Only one instance of ImageProcessing allowed");
    
}
//Calculates Mask from both the contrast pictures. 
cv::Mat ImageProcessing::createMask() {
    cv::Mat mask, sum_contrast_n;
    cv::Mat sum_contrast = (m_contrast[0] + m_contrast[1])/2;

    // minMaxloc data for sum_contrast
    minmaxloc data{ get_minmaxloc(sum_contrast) };
    //std::cout << data;
    // Threshold
    cv::threshold(sum_contrast, mask, 0.5 * data.maxval, 1, cv::THRESH_BINARY);
    //normalizeAndDisplay(mask);
    // Create Structuring Element for opening&closing
    cv::Mat strucutre = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, strucutre);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, strucutre);

    minmaxloc data1{ get_minmaxloc(mask) };
    //std::cout << data1;
    //cv::normalize(sum_contrast, sum_contrast_n, 0, 255, cv::NORM_MINMAX, CV_32F);

    return mask;


    // Debugging and somehting where i tried to use findcontours from opencv
    /*
    cv::imshow("sumcontrast ", sum_contrast);
    cv::imshow("Treshholded", mask);
    cv::waitKey(0);
    */

    /*
    std::pair<cv::Mat, cv::Mat> gradient_horizontal{ grad_strength(m_contrast[0]) };
    std::pair<cv::Mat, cv::Mat> gradient_vertical{ grad_strength(m_contrast[1]) };
    
    //Use gradient pictures
    mask = (gradient_horizontal.first + gradient_horizontal.second)/2;
    cv::Mat mask8u;
    cv::normalize(mask, mask8u, 0, 255.0, cv::NORM_MINMAX, CV_8U);
    minmaxloc value_information(get_minmaxloc(mask8u));
    cv::imshow("NormalizedMask", mask8u);
    cv::Mat canny, edges;
    cv::waitKey(0);
    cv::Canny(mask8u, canny, value_information.maxval*0.15, value_information.maxval*0.3 );
    cv::imshow("Canny", canny);

    cv::waitKey(0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int largest_index = 0;
    double largest_area = 0.0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            largest_index = i;
        }
    }

    std::vector<cv::Point> approx;
    cv::approxPolyDP(contours[largest_index], approx, 10, true);

    cv::Mat mask_f = cv::Mat::zeros(canny.size(), CV_8UC1);
    cv::drawContours(mask_f, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(255), cv::FILLED);
    cv::imshow("Hopefully Contour", mask_f);
    cv::waitKey(0);
    */
    /*Debugging*/
    /*
    cv::imshow("fringeHorizontal_fx", gradient_horizontal.first);
    cv::imshow("fringeHorizontal_fy", gradient_horizontal.second);
    cv::imshow("fringevertical_fx", gradient_vertical.first);
    cv::imshow("fringevertical_fy", gradient_vertical.second);
    cv::waitKey();
    */
}



auto normalizeAndDisplay = [](const cv::Mat& img) -> void {
    cv::Mat norm;
    cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Normalized", norm);
    cv::waitKey(0);
    };

// Function that takes two images. If Mask image is != the original value is saved. Else it is 0.
// This function can be templated
// When the double value is != 0 all values that not masked are shifted about this value. 
// Also in this case the masked values are set to -1!!!!!
cv::Mat ImageProcessing::applyMask(const cv::Mat& mask, const cv::Mat& img, float shift) {
    assert(mask.size() == img.size() && "Mask and image must be of the same size \n");
    assert(img.type() == CV_32FC1 && "Expected float image");

    cv::Mat result = cv::Mat::zeros(img.size(), CV_32FC1);

    for (int row = 0; row < img.rows; ++row) {
        const float* pImg = img.ptr<float>(row);
        const float* pMask = mask.ptr<float>(row);
        float* pRes = result.ptr<float>(row);
        if (!shift) {
            //int this line no shifiting is applied -> all masked values sare set to zero
            for (int col = 0; col < img.cols; ++col) {
                pRes[col] = (pMask[col] != 0.0f) ? pImg[col] : 0.0f;
            }
        }
        else {
            // here the double value != 0, masked pixels are set to -1 and for valid pixel the shift is applied. 
            for (int col = 0; col < img.cols; ++col) {
                *(pRes + col) = (*(pMask + col) != 0.0f) ? (*(pImg + col)+shift) : -1.0f;
            }
        }
    }

    return result;
}

// Shifts all valids pixel about the shift value. IMG(x,y) + shift_value. 
// Just sets all not valid pixels to -1000
void ImageProcessing::shiftStartPhasetoZero(const cv::Mat& mask, cv::Mat& img, float shift_value) {
    assert(mask.size() == img.size());
    assert(img.type() == CV_32FC1 && "Expected float image");

    for (int row = 0; row < img.rows; ++row) {
        float* pImg = img.ptr<float>(row);
        const float* pMask = mask.ptr<float>(row);
        for (int cols = 0; cols < img.cols; ++cols) {
            if (!*(pMask + cols)) {
                *(pImg + cols) = -10.0f;
                continue;
            }
            else if (*(pMask + cols)) {
                *(pImg + cols) -= shift_value;

            }
        }
    }
}

/*
//Just a small example code to do generic programming with iterators
template <class InputIt>
typename std::iterator_traits<InputIt>::difference_type
distance(InputIt first, InputIt last) {
    using category = typename std::iterator_traits<InputIt>::iterator_category;

    if constexpr (std::is_same_v<category, std::random_access_iterator_tag>)
        return last - first;     // fast O(1)
    else {
        typename std::iterator_traits<InputIt>::difference_type n = 0;
        for (; first != last; ++first) ++n;   // slow O(n)
        return n;
    }
}
*/

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


void ImageProcessing::gray_value_calib(std::vector<cv::Mat>& vec) {
    // Create first mean values if needed. 
    assert(!vec.empty() && "If the vector is empty there must be error in acquisition. \n");
    std::vector<cv::Mat> mean_vec{};
    if (runtime_flags.calib.pictures_per_value > 1) {
        std::vector<cv::Mat>::const_iterator begin_mean = vec.cbegin();
        std::vector<cv::Mat>::const_iterator end_mean = advance_return1(begin_mean, runtime_flags.calib.pictures_per_value);
        mean_vec.push_back(mean(std::vector<cv::Mat>(begin_mean, end_mean)));
        for (std::size_t i = 0; i < (std::numeric_limits<uchar>::max() / runtime_flags.calib.stepwidth); ++i) {
            begin_mean = advance_return1(begin_mean, runtime_flags.calib.pictures_per_value);
            end_mean = advance_return1(end_mean, runtime_flags.calib.pictures_per_value);
            mean_vec.push_back(mean(std::vector<cv::Mat>(begin_mean, end_mean)));
        }
    }
    else { mean_vec = std::move(vec); }
    
    // Subtrakt the first image (black) from the brightest (white) the difference is hopefully a usable mask;
    cv::Mat mask = mean_vec.back() - mean_vec.front();
    minmaxloc mask_info{ get_minmaxloc(mask) };
    cv::threshold(mask, mask, mask_info.maxval * 0.5, 255, cv::THRESH_BINARY);
    normalizeAndDisplay(mask);

    std::vector<cv::Scalar_<double>> mean_values;

    for (const auto& mean_img : mean_vec) {
        mean_values.push_back(cv::mean(mean_img, mask));
    }

    m_mean_grayValues = std::move(mean_values);
}



// This is the first implementation of a template elipsis. The function takes a arbitrary ammount of cv::MAts and a function pointer. 
// Becasue the cv::Mat datatype can not be decued at compile time this functions template calls a second inner function where, 
// for the datatypes it is differntiatied. 
template<typename func, typename... Mats>
auto ImageProcessing::forEachPixel(func, Mats&&... mats) {
    static_assert(sizeof...(mats) > 0, "Need at least one matrix."); //compile time check
    auto first std::get<0>(std::forward_as_tuple(std::forward<Mats>(mats)));
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


void ImageProcessing::calc_reproject_error(bool visualizing) {
    assert(runtime_flags.disp.wavelength && "Parameters of the phase pattern are not available. Call generatePattern() before \n");
    m_mask = { createMask() };
    cv::Mat unwrap1_masked, unwrap2_masked;
    unwrap1_masked = applyMask(m_mask, m_unwrapped_phase[0]);
    unwrap2_masked = applyMask(m_mask, m_unwrapped_phase[1]);
    minmaxloc masked_unwrap1{ get_minmaxloc(unwrap1_masked) };
    minmaxloc masked_unwrap2{ get_minmaxloc(unwrap2_masked) };
    std::cout << "Horizontal unwrap " << masked_unwrap1 << '\n' <<
        "Vertical unwrap " << masked_unwrap2 << '\n';

    /*
    //Debugging
    
    normalizeAndDisplay(m_mask);
    normalizeAndDisplay(m_unwrapped_phase[0]);
    normalizeAndDisplay(m_unwrapped_phase[1]);
    normalizeAndDisplay(unwrap1_masked);
    normalizeAndDisplay(unwrap2_masked);
    */

    //Not valid pixels at -10 
    shiftStartPhasetoZero(m_mask, unwrap1_masked, masked_unwrap1.minval);
    shiftStartPhasetoZero(m_mask, unwrap2_masked, masked_unwrap2.minval);

    /*
    minmaxloc shifted1{ get_minmaxloc(unwrap1_masked) };
    minmaxloc shifted2{ get_minmaxloc(unwrap2_masked) };
    std::cout << "Horizontal unwrap " << shifted1 << '\n' <<
        "Vertical unwrap " << shifted2 << '\n';

    normalizeAndDisplay(unwrap1_masked);
    normalizeAndDisplay(unwrap2_masked);
    */

    /*
    cv::cvtColor(m_unwrapped_phase[0], unwrap_rgb, cv::COLOR_GRAY2RGB);
    cv::drawMarker(unwrap_rgb, unwrap1.maxloc, cv::Scalar(255, 0, 0));
    cv::drawMarker(unwrap_rgb, unwrap1.minloc, cv::Scalar(0, 255, 0));
    
    std::pair<cv::Mat, cv::Mat> gradients_horizontal = grad_strength(m_unwrapped_phase[0]);
    std::pair<cv::Mat, cv::Mat> gradients_vertical = grad_strength(m_unwrapped_phase[1]);
    cv::imshow("Gradient horizontal ", gradients_horizontal.first);
    */
    
    // is set to the maximum value
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>> calibrationPoints = 
        generateCalibrationPoints(unwrap1_masked, unwrap2_masked, runtime_flags.disp.wavelength,
            runtime_flags.disp.width, runtime_flags.disp.height);

    
    CV_Assert(!calibrationPoints.first.empty() && calibrationPoints.first.size() == calibrationPoints.second.size());
    CV_Assert(!m_calib_data.cameraMatrix.empty() && !m_calib_data.distCoeffs.empty());

    cv::Mat rvec, tvec;

    /*Calculates Matrix that transforms points from display coordinate system into the camera coordinate system
    X_camera = R*X_Dispaly + t   */
    bool ok = cv::solvePnP(calibrationPoints.second, calibrationPoints.first,
        m_calib_data.cameraMatrix, m_calib_data.distCoeffs, rvec, tvec,
        false, 
        cv::SOLVEPNP_ITERATIVE);
         
    if (!ok) throw std::runtime_error("solvePnP failed.");
    //normalizeAndDisplay(unwrap1_masked);

    // Project Points
    std::vector<cv::Point2f> projected;
    // This function takes the object points. 
    cv::projectPoints(calibrationPoints.second, rvec, tvec, 
        m_calib_data.cameraMatrix, m_calib_data.distCoeffs, projected);

    double sumSq = 0.0;
    double maxErr = 0.0;


    for (size_t i = 0; i < calibrationPoints.first.size(); ++i) {
        double e = cv::norm(calibrationPoints.first[i] - projected[i]); // pixels
        sumSq += e * e;
        maxErr = std::max(maxErr, e);
    }

    double rms = std::sqrt(sumSq / calibrationPoints.first.size());

    std::cout << "Reprojection RMS: " << rms << " px,  max: " << maxErr << " px\n";

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    std::cout << "R =\n" << R << "\n";
    std::cout << "t = " << tvec.t() << " (same units as your objectPoints, here mm)\n";

    // Distance camera↔screen plane (norm of t)
    std::cout << "Camera distance ≈ " << cv::norm(tvec) << " mm\n";

    if (visualizing) {
        // Visualization
        // mainly to see error of the camera calibartion (verzeichnung). We try to draw a line that points in the direction of the error. 
        cv::Mat error_visualizer1;
        cv::cvtColor(m_mask.clone(), error_visualizer1, cv::COLOR_GRAY2BGR);
        for (std::size_t i = 0; i < calibrationPoints.first.size(); ++i) {
            cv::arrowedLine(error_visualizer1, projected[i], calibrationPoints.first[i], cv::Scalar(255, 0, 0), 2);
        }
        cv::imshow("Error mask", error_visualizer1);
        m_reprojection_error_img.push_back(error_visualizer1);

        // mainly to see the error of the grayvalue calibration (periodic errors)
        cv::Mat_<cv::Point_<float>> error_visualizer2(m_mask.size(), cv::Point_<float>(0, 0));
        for (std::size_t i = 0; i < calibrationPoints.first.size(); ++i) {
            cv::Point pt(cvRound(calibrationPoints.first[i].x), cvRound(calibrationPoints.first[i].y));
            if (pt.inside(cv::Rect(0, 0, error_visualizer2.cols, error_visualizer2.rows)))
                error_visualizer2.at<cv::Point2f>(pt) = calibrationPoints.first[i] - projected[i];
        }

        std::vector<cv::Mat> xy(2);
        //Seperate a mask given multiple channels cv::Mat_<cv::Point_<float>> 
        cv::split(error_visualizer2, xy);
        //Another apporach through pointers -> 
        /*
        for (int r = 0; r < points.rows; ++r) {
            const cv::Vec2f* rowPtr = points.ptr<cv::Vec2f>(r);
                for (int c = 0; c < points.cols; ++c) {
                    float x = rowPtr[c][0];
                    float y = rowPtr[c][1];
            }
        }
        */

        minmaxloc x_direc{ get_minmaxloc(xy[0]) };
        minmaxloc y_direc{ get_minmaxloc(xy[1]) };
        double x_range = x_direc.maxval - x_direc.minval;
        double y_range = y_direc.maxval - y_direc.minval;
        int x_max, y_max;
        (x_range > y_range) ? (x_max = 255, y_max = static_cast<int>(255 * (y_range/x_range))) : 
            (x_max = static_cast<int>(255 * (x_range/y_range)), y_max = 255);
        //In this code both images are just normalized to their given range. Therefore the color map gives no information a
        cv::Mat visual_mask = ((xy[0] != 0) | (xy[1] != 0));
        cv::Mat xNorm, yNorm;
        // First normalization, than conversion to the needed Datatype !!!!
        cv::normalize(xy[0], xNorm, 0, x_max, cv::NORM_MINMAX);
        cv::normalize(xy[1], yNorm, 0, y_max, cv::NORM_MINMAX);
        xNorm.convertTo(xNorm, CV_8U);
        yNorm.convertTo(yNorm, CV_8U);

        cv::Mat xcolor, ycolor;
        cv::applyColorMap(xNorm, xcolor, cv::COLORMAP_TURBO);
        cv::applyColorMap(yNorm, ycolor, cv::COLORMAP_TURBO);
        //Set to zero, if the mask is not true. So just extra to set all elements that are masked to zero. 
        xcolor.setTo(cv::Scalar(0, 0, 0), ~visual_mask);
        ycolor.setTo(cv::Scalar(0, 0, 0), ~visual_mask);

        m_reprojection_error_img.push_back(xcolor);
        m_reprojection_error_img.push_back(ycolor);
        cv::imshow("Error x direction", xcolor);
        cv::imshow("Error y direction", ycolor);

        cv::waitKey(0);   
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>
ImageProcessing::generateCalibrationPoints(
    const cv::Mat& unwrapX,
    const cv::Mat& unwrapY,
    float pixelsPer2pi,
    int gridX,
    int gridY,
    float screenWidth_mm,
    float screenHeight_mm,
    float pixel_pitch)
{
    CV_Assert(unwrapX.size() == unwrapY.size());
    CV_Assert(unwrapX.type() == CV_32FC1 && unwrapY.type() == CV_32FC1);
    int screenPxWidth{ runtime_flags.disp.width };
    int screenPxHeight{ runtime_flags.disp.height };
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;

    // Define spacing across image (camera pixels)
    float stepX = static_cast<float>(unwrapX.cols) / (gridX);  //+1
    float stepY = static_cast<float>(unwrapX.rows) / (gridY);  //+1

    //Debug Copy
    cv::Mat debug = unwrapX.clone();

    for (int gy = 0; gy <= gridY; ++gy) {
        //if (stepY == 1) --gy;
        int dy = static_cast<int>(gy * stepY);
        if (dy < 0 || dy >= unwrapX.rows)
            continue;

        const float* row_ptr_X = unwrapX.ptr<float>(dy);
        const float* row_ptr_Y = unwrapY.ptr<float>(dy);
        for (int gx = 0; gx <= gridX; ++gx) {
            //if (stepX == 1) --gx;
            int px = static_cast<int>(gx * stepX);
            //int py = static_cast<int>(gx * stepY);

            if ((*(row_ptr_X + px) == -10.0f) || (*(row_ptr_Y + px) == -10.0f)) continue; // Ye haa -> pointer arithmetic bitches 

            if (px < 0 || dy < 0 || px >= unwrapX.cols || dy >= unwrapX.rows) {
                std::cout << "Something must have went terribly wrong: GenerateCalibrationPoints() \n";
                continue;
            }

            //Can be used to acces array element, but documentatoins says it is slow.
            //float phiX = unwrapX.at<float>(py, px);
            //float phiY = unwrapY.at<float>(py, px);
            // Get the phase value in each pictures for the same pixel if valid. 
            float phiX = (*(row_ptr_X + px));
            float phiY = (*(row_ptr_Y + px));

            // Maximum possible phase values
            float max_phase_value_u = (static_cast<float>(runtime_flags.disp.width) / pixelsPer2pi) * CV_2PI;
            float max_phase_value_v = (static_cast<float>(runtime_flags.disp.height) / pixelsPer2pi) * CV_2PI;

            //Check if phase value is in logical range. These boarder need to be less strict
            if (phiX > max_phase_value_u || phiY > max_phase_value_v) { throw std::runtime_error ("The phase is not within allowed bounds. "); }

            // Convert unwrapped phase  screen coordinates in subpixel (float, float) values. 
            // Be carefull -> the phase direction is: <- (left for horizontal) and î (upwards for vertical)
            // The coordinate system should be in top left so we have to convert back.  
            float u_s = (phiX / (2.0f * CV_PI)) * pixelsPer2pi;
            float v_s = (phiY / (2.0f * CV_PI)) * pixelsPer2pi;

            u_s = screenPxWidth - u_s;
            v_s = screenPxHeight - v_s;

            // Convert to millimetres using screen dimensions
            float Xs = u_s * pixel_pitch;
            float Ys = v_s * pixel_pitch;

            // Store results
            imagePoints.emplace_back(static_cast<float>(px), static_cast<float>(dy));
            objectPoints.emplace_back(Xs, Ys, 0.0f);
            //cv::drawMarker(debug, cv::Point(px,dy), cv::Scalar(0));
        } 
    }
    normalizeAndDisplay(debug);
    return { imagePoints, objectPoints };
}

void ImageProcessing::load_calib(std::string path) {
    m_calib_data = getfromFile(path);
}


void ImageProcessing::load_frames(const std::string& path) {
    try {
        //Checks only for the first element in arrays. It is assumed when the first is empty the second one must be to. 
        if (m_s1[0].empty() && m_s2[0].empty() && m_s3[0].empty() && m_baseIntensity[0].empty() &&
            m_contrast[0].empty() && m_phase[0].empty() && m_wrapped_phase[0].empty() && m_unwrapped_phase[0].empty()) {
            create();
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
        }
        else { std::runtime_error e("The image container already have data in it. This is not allowed. \n"); }

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


void ImageProcessing::saveImages(std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    
    if (fs.isOpened()) {
        fs << "wrappedPhasehorizontal" << m_wrapped_phase.at(0);
        fs << "wrappedPhasevertical" << m_wrapped_phase.at(1);
        fs << "contrasthorizontal" << m_contrast.at(0);
        fs << "contrastvertical" << m_contrast.at(1);
        fs << "baseIntensityhorizontal" << m_baseIntensity.at(0);
        fs << "baseIntensityvertical" << m_baseIntensity.at(1);
        fs << "unwrappedPhasehorizontal" << m_unwrapped_phase.at(0);
        fs << "unwrappedPhasevertical" << m_unwrapped_phase.at(1);
    }
}

ImageProcessing::~ImageProcessing() {
    --instance_counter;
}
void ImageProcessing::wrapped_phase() {
    
    create(m_frames);
	std::cout << "Datatype " << m_frames[0].type() << '\n';
    
    if (runtime_flags.get_number_of_pictures_per_pattern() <= 0) {
        std::cerr << "Flag how many Picutres per Pattern are created must be set to specific value != 0 \n";
        return;
    }

    if (runtime_flags.get_number_of_pictures_per_pattern() > 1) {
        int n_expected_frames = runtime_flags.get_number_of_pictures_per_pattern() *
            runtime_flags.get_number_of_shifts() * 2;
        std::cout << "Expected " << n_expected_frames << '\n' <<
            "std::vector size " << m_frames.size() << '\n';

        assert(n_expected_frames == m_frames.size() && "For valid Processing, number of expected Frames (calculated from flagHandler.hpp\
			 flags) must match vector size \n");
        std::vector<cv::Mat>::iterator begin = m_frames.begin();
        
        for (int i = 0; i < (runtime_flags.get_number_of_shifts()*2); ++i) {
            std::vector<cv::Mat>::iterator end = begin + runtime_flags.get_number_of_pictures_per_pattern();
            m_raw_phase.push_back(mean(std::vector<cv::Mat>(begin, end)));
            begin = end;
        }
    }

    if (runtime_flags.get_number_of_pictures_per_pattern() == 1) {
        m_raw_phase = m_frames;
    }
    
    for (auto& frame : m_raw_phase) {
        if (frame.type() != CV_32F)
            frame.convertTo(frame, CV_32F, 1.0f / 255.0f);
    }

    for (int i = 0; i < runtime_flags.get_number_of_shifts(); ++i) {    
        float phase = static_cast<float>((CV_2PI * i) / runtime_flags.get_number_of_shifts());
        //m_s1.at(0) += m_raw_phase.at(i).forEach<float>([&](float& a, const int* position) -> void {
        //    a = a * std::sin(phase); });
        m_s1.at(0) += m_raw_phase.at(i) * std::sinf(phase);
        m_s1.at(1) += m_raw_phase.at(i + runtime_flags.get_number_of_shifts()) * std::sinf(phase);
        m_s2.at(0) += m_raw_phase.at(i) * std::cosf(phase);
        m_s2.at(1) += m_raw_phase.at(i + runtime_flags.get_number_of_shifts()) * std::cosf(phase);
        m_s3.at(0) += m_raw_phase.at(i);
        m_s3.at(1) += m_raw_phase.at(i + runtime_flags.get_number_of_shifts());
    }

    for (int i = 0; i < m_baseIntensity.size(); ++i) {
        m_baseIntensity.at(i) = m_s3.at(i) / runtime_flags.get_number_of_shifts();
        //Initialize already a array with the right datatype and size()
    }

    for (int i = 0; i < m_s1.at(0).rows; ++i) {
        for (int j = 0; j < m_s1.at(0).cols; ++j) {
            m_wrapped_phase.at(0).at<float>(i, j) = std::atan2f(m_s1.at(0).at<float>(i, j), m_s2.at(0).at<float>(i, j));
            m_wrapped_phase.at(1).at<float>(i, j) = std::atan2f(m_s1.at(1).at<float>(i, j), m_s2.at(1).at<float>(i, j));
            m_contrast.at(0).at<float>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(0).at<float>(i, j), 2) + std::pow(m_s2.at(0).at<float>(i, j), 2))) / m_s3.at(0).at<float>(i, j);
            m_contrast.at(1).at<float>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(1).at<float>(i, j), 2) + std::pow(m_s2.at(1).at<float>(i, j), 2))) / m_s3.at(1).at<float>(i, j);
        }
    }
    
    double phaseminVal, phasemaxVal, baseminVal, basemaxval, conminval, conmaxval;
    cv::Point phaseminloc, phasemaxloc, baseminloc, basemaxloc, conminloc, conmaxloc;
    
    /*Debugging*/
    
    cv::minMaxLoc(m_wrapped_phase[1],&phaseminVal, &phasemaxVal);
    cv::minMaxLoc(m_baseIntensity[1], &baseminVal, &basemaxval);
    cv::minMaxLoc(m_contrast[1], &conminval, &conmaxval);

    std::cout << "Phase: " << phaseminVal << " … " << phasemaxVal << '\n'
        << "Base intensity: " << baseminVal << " … " << basemaxval << '\n'
        << "Contrast: " << conminval << " … " << conmaxval << '\n';

    cv::Mat wrapped8cu1, wrapped8cu2, constrast8cu1, contrast8cu2;
    cv::normalize(m_wrapped_phase[0], wrapped8cu1, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_wrapped_phase[1], wrapped8cu2, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_contrast[0], constrast8cu1, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_contrast[1], contrast8cu2, 0, 255, cv::NORM_MINMAX, CV_8U);
    std::filesystem::path outDir("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out");
    if(std::filesystem::exists(outDir) == false) {
        std::filesystem::create_directory(outDir);
	}
	std::filesystem::path path1 = outDir / "wrapped_phase1.png";
	std::filesystem::path path2 = outDir / "wrapped_phase2.png";
	std::filesystem::path path3 = outDir / "base_intensity1.png";
	std::filesystem::path path4 = outDir / "contrast1.png";
	std::filesystem::path path5 = outDir / "base_intensity2.png";
	std::filesystem::path path6 = outDir / "contrast2.png";

    try {
        cv::imwrite(path1.string(), wrapped8cu1);
        cv::imwrite(path2.string(), wrapped8cu2);
		cv::imwrite(path4.string(), constrast8cu1);
        cv::imwrite(path6.string(), contrast8cu2);

    }
    catch (std::exception& e) { std::cout << "Excpetion " << e.what() << std::endl; }
    
    //Debugging
    /*
    auto [Hx, Hy] = grad_strength(m_wrapped_phase[0]); // horizontal set
    auto [Vx, Vy] = grad_strength(m_wrapped_phase[1]); // vertical set
    std::cout << "Wrapped H: |dx|=" << Hx << " |dy|=" << Hy << "\n";
    std::cout << "Wrapped V: |dx|=" << Vx << " |dy|=" << Vy << "\n";
    */

    /*
    cv::imshow("Contrast", m_contrast.at(0));
    cv::imshow("Base Intensity", m_baseIntensity.at(0));
    cv::imshow("Phase", m_phase.at(0));

    cv::Mat phaseVis, contrastVis;
    cv::normalize(m_phase.at(0), phaseVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_contrast.at(0), contrastVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::applyColorMap(phaseVis, phaseVis, cv::COLORMAP_JET);
    cv::applyColorMap(contrastVis, contrastVis, cv::COLORMAP_JET);

    cv::imshow("Phase (color)", phaseVis);
    cv::imshow("Contrast (color)", contrastVis);
    cv::waitKey(0);
    */
    /*Just to see results*/
    /*
    std::thread img(&ImageHandler::run, imgHandler, 3);
    imgHandler.imshow_Camera(m_baseIntensity.at(0));
    imgHandler.imshow_Pattern(m_phase.at(0));
    imgHandler.imshow_Processed(m_contrast.at(0));
    std::this_thread::sleep_for(std::chrono::seconds(10));
    imgHandler.stop();
    if (img.joinable()) img.join();
    */
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

    /*
    goldsteinUnwrapCV(m_phase.at(0), m_unwrapped_phase.at(0));
	cv::imshow("Goldstein Unwrapped Phase", m_unwrapped_phase.at(0));
	goldsteinUnwrapCV(m_phase.at(1), m_unwrapped_phase.at(1));
	cv::imshow("Goldstein Unwrapped Phase 2", m_unwrapped_phase.at(1));
	cv::waitKey(0);
    
    double phaseminVal, phasemaxVal, baseminVal, basemaxval, conminval, conmaxval;

    cv::minMaxLoc(m_unwrapped_phase[0], &phaseminVal, &phasemaxVal);
    cv::minMaxLoc(m_unwrapped_phase[1], &baseminVal, &basemaxval);

    std::cout << "Phase1: " << phaseminVal << " … " << phasemaxVal << '\n'
        << "Phase 2: " << baseminVal << " … " << basemaxval << '\n';
        */
}


void ImageProcessing::unwrapped_phase() {
    //std::cout << "Phase unwrapping \n";
    cv::Mat unwrappedPhase1, unwrappedPhase2;
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params params;
    params.height = runtime_flags.pixel_y;
    params.width = runtime_flags.pixel_x;
    params.histThresh = CV_PI / 10; // you can tune this threshold
    params.nbrOfSmallBins = 20;
    params.nbrOfLargeBins = 10;
    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> unwrapping = cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);
    cv::Mat mask1 = (m_contrast.at(0) > 0.1f);  // choose threshold as needed
	cv::Mat mask2 = (m_contrast.at(1) > 0.1f);  // choose threshold as needed
    unwrapping->unwrapPhaseMap(m_wrapped_phase.at(0), m_unwrapped_phase.at(0), mask1);

    // vertical unwrap, rotated to horizontal orientation
    cv::Mat wrappedTransposed, maskTransposed, unwrappedTransposed;
    cv::transpose(m_wrapped_phase.at(1), wrappedTransposed);
    cv::transpose(m_contrast.at(1), maskTransposed);
    maskTransposed = (maskTransposed > 0.3f);

    // create the same params (same pixel_y/pixel_x, do NOT swap)
    params.height = wrappedTransposed.rows;
    params.width = wrappedTransposed.cols;
    params.histThresh = CV_PI / 10;
    params.nbrOfSmallBins = 20;
    params.nbrOfLargeBins = 10;

    auto unwrapping1 = cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);

    // unwrap the *transposed* image
    unwrapping1->unwrapPhaseMap(wrappedTransposed, unwrappedTransposed, maskTransposed);

    // transpose back
    cv::transpose(unwrappedTransposed, unwrappedPhase2);
    m_unwrapped_phase.at(1) = unwrappedPhase2;
    cv::Mat unwrappedPhase1_8u, unwrappedPhase2_8u;

    cv::normalize(m_unwrapped_phase.at(0), unwrappedPhase1_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_unwrapped_phase.at(1), unwrappedPhase2_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imshow("Unwrapped Phase 1", unwrappedPhase1_8u);
    cv::imshow("Unwrapped Phase 2", unwrappedPhase2_8u);
    cv::waitKey(0);
    /*
    std::filesystem::path outDir("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out");
    if (std::filesystem::exists(outDir) == false) {
        std::filesystem::create_directory(outDir);
    }
    std::filesystem::path path1 = outDir / "unwrapped_phase1.png";
    std::filesystem::path path2 = outDir / "unwrapped_phase2.png";
    cv::imwrite(path1.string(), unwrappedPhase1_8u);
    cv::imwrite(path2.string(), unwrappedPhase2_8u);
    */
}

cv::Mat ImageProcessing::mean(std::vector<cv::Mat>& vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            std::cerr << "All pictures must be same kind and type. \n";
            throw std::runtime_error ("All pictures must be same kind and type. \n");
        }
    }
    cv::Mat acc;
    vec[0].convertTo(acc, CV_32FC1);

    for (size_t i = 1; i < vec.size(); ++i) {
        cv::Mat temp;
        vec[i].convertTo(temp, CV_32FC1);
        acc += temp;   // pixelweise Addition
    }

    acc /= static_cast<float>(vec.size());  // pixelweise Division

    // Zurück zu 8 Bit
    cv::Mat average;
    acc.convertTo(average, CV_8UC1);
    //Debugging
	//cv::imshow("Mean Image", average);
    //cv::waitKey();
    return average;
}


cv::Mat ImageProcessing::load_images(std::string path) {
    std::filesystem::path image_location{ path };
    if (std::filesystem::exists(image_location)) {
        return cv::imread(path, cv::ImreadModes::IMREAD_GRAYSCALE);
    }
}

