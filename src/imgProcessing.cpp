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

void ImageProcessing::unwrap_row(const cv::Mat& wrapped, cv::Mat& unwrapped, const cv::Mat& mask, int row)
{
    double two_pi = CV_2PI;

    double prev = 0;
        
    double k = 0;

    int valid_counter{0};

    for (int col = 0; col < wrapped.cols; ++col)
    {
        if (!mask.at<uchar>(row, col))
            continue;
        
        double current = wrapped.at<double>(row, col);
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



void ImageProcessing::manual_phaseUnwrap() {
    CV_Assert(!m_wrapped_phase.empty());
    //m_mask = createMask();

    m_mask = cv::Mat(m_wrapped_phase[0].rows, m_wrapped_phase[0].cols, CV_8U, cv::Scalar(255));
    
    for (std::size_t count = 0; count < m_wrapped_phase.size(); count++) {
        
        cv::Mat wrapped64;
        m_wrapped_phase[count].convertTo(wrapped64, CV_64F);

        // Initialize output
        m_unwrapped_phase[count] = cv::Mat::zeros(wrapped64.size(), CV_64F);

        if (count == 0)
        {
            // horizontal unwrap
            for (int row = 0; row < wrapped64.rows; ++row)
                unwrap_row(wrapped64, m_unwrapped_phase[count], m_mask, row);
        }
        else if (count == 1)
        {
            // vertical unwrap
            for (int col = 0; col < wrapped64.cols; ++col)
                unwrap_column(wrapped64, m_unwrapped_phase[count], m_mask, col);
        }

    }
    //normalizeAndDisplay(m_mask);
    //normalizeAndDisplay(m_unwrapped_phase[0]);
    //normalizeAndDisplay(m_unwrapped_phase[1]);
   
    cv::Mat unwrapchar0, unwrapchar1;
    cv::normalize(m_unwrapped_phase[0], unwrapchar0, 0, 255, cv::NORM_MINMAX);
    cv::normalize(m_unwrapped_phase[1], unwrapchar1, 0, 255, cv::NORM_MINMAX);

    unwrapchar0.setTo(cv::Scalar(0), ~m_mask);
    unwrapchar1.setTo(cv::Scalar(0), ~m_mask);

    //normalizeAndDisplay(unwrapchar0);
    //normalizeAndDisplay(unwrapchar1);

    cv::imwrite("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/Unwrap1.png", unwrapchar0);
    cv::imwrite("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/Unwrap2.png", unwrapchar1);
}


//Calculates Mask from both the contrast pictures. 
cv::Mat ImageProcessing::createMask() {
    cv::Mat mask, sum_contrast_n;
    assert((m_contrast[0].type() == CV_32F) || m_contrast[0].type() == CV_64F);
    cv::Mat sum_contrast = (m_contrast[0] + m_contrast[1])/2;

    // minMaxloc data for sum_contrast
    minmaxloc data{ get_minmaxloc(sum_contrast) };
    //std::cout << data;
    // Threshold
    cv::threshold(sum_contrast, mask, 0.2 * data.maxval, 1, cv::THRESH_BINARY);
    //normalizeAndDisplay(mask);
    // Create Structuring Element for opening&closing
    cv::Mat strucutre = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, strucutre, cv::Point2d(-1, -1), 2);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, strucutre, cv::Point2d(-1, -1), 2);

    minmaxloc data1{ get_minmaxloc(mask) };
    //std::cout << data1;
    //cv::normalize(sum_contrast, sum_contrast_n, 0, 255, cv::NORM_MINMAX, CV_32F);
    cv::Mat mask_8u;
    cv::normalize(mask, mask_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    //normalizeAndDisplay(mask_8u);
    return mask_8u;


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


void ImageProcessing::gray_value_calib(std::vector<cv::Mat>& vec) {
    // Create first mean values if needed. 
    assert(!vec.empty() && "If the vector is empty there must be error in acquisition. \n");
    std::vector<cv::Mat> mean_vec{};
    if (runtime_flags.calib.pictures_per_value > 1) {
        std::vector<cv::Mat>::const_iterator begin_mean = vec.cbegin();
        
        std::vector<cv::Mat>::const_iterator end_guard = vec.end();

        for (std::size_t i = 0; i < (std::numeric_limits<uchar>::max() / runtime_flags.calib.stepwidth); ++i) {
            std::vector<cv::Mat>::const_iterator end_mean = advance_return1(begin_mean, runtime_flags.calib.pictures_per_value);
            CV_Assert(end_mean <= end_guard && "Iterator dereference element out of bounds");
            
            mean_vec.push_back(mean(std::vector<cv::Mat>(begin_mean, end_mean)));
            begin_mean = end_mean;
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
        

        auto repro2to3error = project2to3d(caliPoints, m_calib_data.cameraMatrix,
            m_calib_data.distCoeffs, rvec, tvec);

        std::cout << "Caluculated the float path \n";
        // Stored as std::variants< ... <float>, ... <double>>
        m_calib_points = caliPoints;
        m_repro_error = repro2to3error;
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
        
        auto repro2to3error = project2to3d(caliPoints, m_calib_data.cameraMatrix,
            m_calib_data.distCoeffs, rvec, tvec);

        std::cout << "Caluculated the double path \n";
        // Stored as std::variants< ... <float>, ... <double>>
        m_calib_points = caliPoints;
        m_repro_error = repro2to3error;
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

void ImageProcessing::wrapped_phase() {
    std::cout << "Vector size " << m_frames.size() << '\n';
    // create(m_frames);
	std::cout << "Datatype " << m_frames[0].type() << '\n';
    
    if (runtime_flags.phase_shift.n_pics_per_Phase <= 0) {
        std::cerr << "Flag how many Picutres per Pattern are created must be set to specific value != 0 \n";
        return;
    }

    if (runtime_flags.phase_shift.n_pics_per_Phase > 1) {
        int n_expected_frames = runtime_flags.phase_shift.n_pics_per_Phase *
            runtime_flags.phase_shift.n_shifts * 2;
        std::cout << "Expected " << n_expected_frames << '\n' <<
            "std::vector size " << m_frames.size() << '\n';

        assert(n_expected_frames == m_frames.size() && "For valid Processing, number of expected Frames (calculated from flagHandler.hpp\
			 flags) must match vector size \n");
        std::vector<cv::Mat>::iterator begin = m_frames.begin();
        
        for (int i = 0; i < (runtime_flags.phase_shift.n_shifts *2); ++i) {
            std::vector<cv::Mat>::iterator end = begin + runtime_flags.phase_shift.n_pics_per_Phase;
            // Constructs a vector with the contents of the range [first, last). Each iterator in [first, last) is dereferenced exactly once.
            m_raw_phase.push_back(mean(std::vector<cv::Mat>(begin, end)));
            begin = end;
        }
    }

    if (runtime_flags.phase_shift.n_pics_per_Phase == 1) {
        m_raw_phase = m_frames;
    }
    
    //current_type 1= float, 2 = double
    for (auto& frame : m_raw_phase) {
        if ((static_cast<int>(current_type) == 1) &&
            frame.type() != CV_32F) {
            frame.convertTo(frame, CV_32F, 1.0f / 255.0f);
        }
        if ((static_cast<int>(current_type) == 2) &&
            frame.type() != CV_64F) {
            frame.convertTo(frame, CV_64F, 1.0 / 255.0);
        }
    }

    // different process for the datatypes. 
    switch (static_cast<int>(current_type)) {
    case(1):
        for (int i = 0; i < runtime_flags.phase_shift.n_shifts; ++i) {
            float phase = static_cast<float>((CV_2PI * i) / static_cast<float>(runtime_flags.phase_shift.n_shifts));
            //m_s1.at(0) += m_raw_phase.at(i).forEach<float>([&](float& a, const int* position) -> void {
            //    a = a * std::sin(phase); });
            m_s1.at(0) += m_raw_phase.at(static_cast<std::size_t>(i)) * std::sin(phase);
            m_s1.at(1) += m_raw_phase.at(static_cast<std::size_t>(i + runtime_flags.phase_shift.n_shifts)) * std::sin(phase);
            m_s2.at(0) += m_raw_phase.at(static_cast<std::size_t>(i)) * std::cos(phase);
            m_s2.at(1) += m_raw_phase.at(static_cast<std::size_t>(i + runtime_flags.phase_shift.n_shifts)) * std::cos(phase);
            m_s3.at(0) += m_raw_phase.at(static_cast<std::size_t>(i));
            m_s3.at(1) += m_raw_phase.at(static_cast<std::size_t>(i + runtime_flags.phase_shift.n_shifts));
        }

        for (int i = 0; i < m_s1.at(static_cast<std::size_t>(0)).rows; ++i) {
            for (int j = 0; j < m_s1.at(static_cast<std::size_t>(0)).cols; ++j) {
                m_wrapped_phase.at(0).at<float>(i, j) = std::atan2f(m_s1.at(0).at<float>(i, j), m_s2.at(0).at<float>(i, j)); 
                m_wrapped_phase.at(1).at<float>(i, j) = std::atan2f(m_s1.at(1).at<float>(i, j), m_s2.at(1).at<float>(i, j));
                m_contrast.at(0).at<float>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(0).at<float>(i, j), 2) + std::pow(m_s2.at(0).at<float>(i, j), 2))) / m_s3.at(0).at<float>(i, j);
                m_contrast.at(1).at<float>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(1).at<float>(i, j), 2) + std::pow(m_s2.at(1).at<float>(i, j), 2))) / m_s3.at(1).at<float>(i, j);
            }
        }
        break;
    case(2):
        for (int i = 0; i < runtime_flags.phase_shift.n_shifts; ++i) {
            double phase = static_cast<double>((CV_2PI * i) / static_cast<double>(runtime_flags.phase_shift.n_shifts));
            //m_s1.at(0) += m_raw_phase.at(i).forEach<float>([&](float& a, const int* position) -> void {
            //    a = a * std::sin(phase); });
            m_s1.at(0) += m_raw_phase.at(static_cast<std::size_t>(i)) * std::sin(phase);
            m_s1.at(1) += m_raw_phase.at(static_cast<std::size_t>(i + runtime_flags.phase_shift.n_shifts)) * std::sin(phase);
            m_s2.at(0) += m_raw_phase.at(static_cast<std::size_t>(i)) * std::cos(phase);
            m_s2.at(1) += m_raw_phase.at(static_cast<std::size_t>(i + runtime_flags.phase_shift.n_shifts)) * std::cos(phase);
            m_s3.at(0) += m_raw_phase.at(static_cast<std::size_t>(i));
            m_s3.at(1) += m_raw_phase.at(static_cast<std::size_t>(i + runtime_flags.phase_shift.n_shifts));
        }

        for (int i = 0; i < m_s1.at(static_cast<std::size_t>(0)).rows; ++i) {
            for (int j = 0; j < m_s1.at(static_cast<std::size_t>(0)).cols; ++j) {
                m_wrapped_phase.at(0).at<double>(i, j) = std::atan2(m_s1.at(0).at<double>(i, j), m_s2.at(0).at<double>(i, j)); // Be carefull atan2 is inverted
                m_wrapped_phase.at(1).at<double>(i, j) = std::atan2(m_s1.at(1).at<double>(i, j), m_s2.at(1).at<double>(i, j));
                m_contrast.at(0).at<double>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(0).at<double>(i, j), 2) + std::pow(m_s2.at(0).at<double>(i, j), 2))) / m_s3.at(0).at<double>(i, j);
                m_contrast.at(1).at<double>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(1).at<double>(i, j), 2) + std::pow(m_s2.at(1).at<double>(i, j), 2))) / m_s3.at(1).at<double>(i, j);
            }
        }
        break;
    default: {
        throw std::exception("Wrapped Phase datatype is wrong");
        }
    }

    for (std::size_t i = 0; i < m_baseIntensity.size(); ++i) {
        m_baseIntensity.at(i) = m_s3.at(i) / runtime_flags.phase_shift.n_shifts;
        //Initialize already a array with the right datatype and size()
    }
    /*Debugging*/

    /*normalizeAndDisplay(m_baseIntensity[0]);
    normalizeAndDisplay(m_contrast[0]);
    normalizeAndDisplay(m_wrapped_phase[0]);*/
    
    /*
    minmaxloc wrapped1{ get_minmaxloc(m_wrapped_phase[0])};
    minmaxloc wrapped2{ get_minmaxloc(m_wrapped_phase[1]) };
    minmaxloc contrast1{ get_minmaxloc(m_contrast[0]) };
    
    std::cout << wrapped1 << wrapped2 << contrast1 << '\n';
    */

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
    params.height = runtime_flags.camera_data.pixel_y;
    params.width = runtime_flags.camera_data.pixel_x;
    params.histThresh = CV_PI / 10; // you can tune this threshold
    params.nbrOfSmallBins = 20;
    params.nbrOfLargeBins = 10;
    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> unwrapping = cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);
    m_mask = createMask();  // choose threshold as needed
    //normalizeAndDisplay(m_mask);
    //normalizeAndDisplay(m_wrapped_phase[0]);
    for (auto& m : m_wrapped_phase) {
        std::cout << "cv::unwrapPhaseMap expects float images -> convert double to float for unwrap!\n";
        m.convertTo(m, CV_32F);
    }
    std::cout << "mask size " << m_mask.size() << '\n' <<
        "unwrap size " << m_wrapped_phase.at(0).size() << '\n';

    if (m_mask.empty() || m_mask.type() != CV_8U) {
        std::cerr << "Invalid mask — converting.\n";
        if (!m_mask.empty())
            m_mask.convertTo(m_mask, CV_8U, 255.0);
        else
            m_mask = cv::Mat::ones(m_wrapped_phase[0].size(), CV_8U);
    }
    for (auto& m : m_unwrapped_phase) {
        if(m.type() == CV_64F)
            m.convertTo(m, CV_32F);
    }
    minmaxloc mask_data{ get_minmaxloc(m_mask) };
    cv::Mat unwrap_mask;
    if (mask_data.minval == 0 && mask_data.maxval == 1) cv::Mat unwrap_mask = m_mask * 255;
    else unwrap_mask = m_mask;
    //cv::setBreakOnError(true);
    //cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
    CV_Assert(unwrap_mask.size() == m_wrapped_phase.at(0).size());
    unwrapping->unwrapPhaseMap(m_wrapped_phase.at(0), m_unwrapped_phase.at(0) , unwrap_mask);
    //unwrapped1
    // vertical unwrap, rotated to horizontal orientation
    cv::Mat wrappedTransposed, maskTransposed, unwrappedTransposed;
    cv::transpose(m_wrapped_phase.at(1), wrappedTransposed);
    cv::transpose(m_contrast.at(1), maskTransposed);
    maskTransposed = unwrap_mask.t();

    // create the same params (same pixel_y/pixel_x, do NOT swap)
    params.height = runtime_flags.camera_data.pixel_x;
    params.width = runtime_flags.camera_data.pixel_y;
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


    if (static_cast<int>(current_type) == 1) {
        for (auto& m : m_unwrapped_phase) {
            if (m.type() != CV_32F) m.convertTo(m, CV_32F);
        }
        for (auto& m : m_wrapped_phase) {
            if (m.type() != CV_32F) m.convertTo(m, CV_32F);
        }
    }
    else if (static_cast<int>(current_type) == 2) {
        for (auto& m : m_unwrapped_phase) {
            if (m.type() != CV_64F) m.convertTo(m, CV_64F);
        }
        for (auto& m : m_wrapped_phase) {
            if (m.type() != CV_64F) m.convertTo(m, CV_64F);
        }
    }

    //normalizeAndDisplay(m_unwrapped_phase[0]);
    //normalizeAndDisplay(m_unwrapped_phase[1]);
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

