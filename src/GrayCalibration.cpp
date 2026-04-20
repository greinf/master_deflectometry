#include "GrayCalibration.hpp"
#include "GrayCalibration_Utils.hpp"
#include "cassert"
#include <memory>
#include "imageStore.hpp"
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>
#include "utils.hpp"
#include <ceres/ceres.h>
#include <cmath>


struct GrayCalibration::Impl {
    struct GrayLut {
        std::array<std::pair<double, double>, 256> gray_Lut{};
        double m_min_LUT{};
        double m_max_LUT{};
        double m_scale_factorLUT{};
        double m_biasLut{};
    } gray_LUT{};

    // Active
	std::vector<cv::Mat> modelBias_a{};
	std::vector<cv::Mat> model_a{};

    // Passive
    std::vector<cv::Mat> modelBias_b{};
    std::vector<cv::Mat> model_b{};

    template<typename T>
    bool empty(const T) {
        static_assert(std::is_base_of<_Base::specifier_Base, T>::value);
        if constexpr (std::is_same<T, GrayCalibration_specifier::Active::LUT>::value ||
            std::is_same<T, GrayCalibration_specifier::Passive::LUT>::value) {
            return LutEmpty(gray_LUT.gray_Lut);
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Active::Model>::value) {
            return model_a.empty();
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Active::Model_Bias>::value) {
            return modelBias_a.empty();
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Passive::Model>::value) {
            return model_b.empty();
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Passive::Model_Bias>::value) {
            return modelBias_b.empty();
        }
    }
};


namespace ceresCost {
    struct ExponentialResidual {
        ExponentialResidual(double x, double y)
            : x_(x), y_(y) { }

        template <typename T>
        bool operator()(
            const T* const i_max,
            const T* const gamma,
            const T* const i_0,
            T* residual) const {
            residual[0] = (T)y_ - (*i_max * ceres::pow((T)x_, *gamma) + *i_0);
            return true;
        }

    private:
        const double x_;
        const double y_;
    };
}


namespace detail {

    inline bool isFinite(double x) { return std::isfinite(x); }

    inline double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(v, hi));
    }

    struct Sample {
        double u{}; // g/255
        double I{}; // measured intensity
    };


    // Try to do some data sampling before we start 
    inline std::vector<Sample> buildSamples(const std::array<double, 256>& measured,
        double sat_cut_rel,
        double eps)
    {
        // If the biggest Element is smaller than epsion than Abort directly
        double Imax_obs = *std::max_element(measured.begin(), measured.end());
        if (Imax_obs < eps) return{};
            //throw std::runtime_error("No valid signal (Imax_obs ~ 0).");

        std::vector<Sample> samples;
        samples.reserve(256);

        for (int g = 0; g <= 255; ++g) {
            double I = measured[g];
            // We cut first if the smaller than epsion.
            if (I < eps) continue;

            // normalize to value 0 ... 1
            double u = static_cast<double>(g) / 255.0;
            if (u <= 0.0) continue;

            // Save the intensity Val 0 ... 255 I and normalized to 0 ... 1
            samples.push_back({ u, I });
        }
        if (samples.size() < 80) return {}; 
        return samples;
    }

    inline double predict(double Imax, double gamma, double u, double i_0 = 0) {
        return Imax * std::pow(u, gamma) + i_0;
    }

    inline double computeSSE(const std::vector<Sample>& s, double Imax, double gamma) {
        double sse = 0.0;
        for (const auto& p : s) {
            double yhat = predict(Imax, gamma, p.u);
            double e = p.I - yhat;
            sse += e * e;
        }
        return sse;
    }

    // compute R2 on intensity values
    inline double computeR2(
        const std::vector<Sample>& s, 
        double Imax, 
        double gamma, 
        double eps,
        double Imin = 0) {
        double meanI = 0.0;
        for (auto& p : s) meanI += p.I;
        meanI /= std::max<size_t>(1, s.size());

        double sst = 0.0, sse = 0.0;
        for (auto& p : s) {
            double yhat = predict(Imax, gamma, p.u, Imin);
            double e = p.I - yhat;
            sse += e * e;

            double d = p.I - meanI;
            sst += d * d;
        }
        if (!(sst > eps)) return 0.0;
        return 1.0 - sse / sst;
    }
} // namespace detail

void GrayCalibration::updateLut(
    const std::array<std::pair<double, double>, 256> LUT)
{
    m_impl->gray_LUT.gray_Lut = LUT;
}

void GrayCalibration::updateModelActive(
    const std::vector<cv::Mat> images)
{
    m_impl->model_a = images;
}

void GrayCalibration::updateModelPassive(
    const std::vector<cv::Mat> images)
{
    m_impl->model_b = images;
}

void GrayCalibration::updateModel_BiasActive(
    const std::vector<cv::Mat> images)
{
    m_impl->modelBias_a = images;
}

void GrayCalibration::updateModel_BiasPassive(
    const std::vector<cv::Mat> images)
{
    m_impl->modelBias_b = images;
}

void GrayCalibration::smoothModelImages(
    std::vector<cv::Mat>& images,
    int kernel_size)
{
    CV_Assert(kernel_size > 0);
    CV_Assert(kernel_size % 2 == 1);

    cv::Mat k = cv::Mat::ones(kernel_size, 1, CV_64F);
    k /= cv::sum(k)[0];

    for (auto& img : images) {
        cv::Mat out;
        cv::sepFilter2D(img, out, CV_64F, k, k, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
        img = out;
    }
}

cv::Mat GrayCalibration::applyCalibration(
    const GrayCalibration_specifier::Active::LUT spec,
    const cv::Mat& image,
    cv::Mat& mask)
{
    CV_Assert(!image.empty());
    CV_Assert(image.channels() == 1);
    CV_Assert(m_impl != nullptr);

    cv::Mat image64;
    if (image.type() != CV_64F) image.convertTo(image64, CV_64F);
    else image64 = image;

    // Cause of the limited values
    cv::Mat img_scaled = image64 * m_impl->gray_LUT.m_scale_factorLUT;
    img_scaled = img_scaled + m_impl->gray_LUT.m_biasLut ;

    if (mask.empty()) mask = cv::Mat::ones(image.size(), CV_8U);
    return applyLut(img_scaled, mask);
}

cv::Mat GrayCalibration::applyCalibration(
    const GrayCalibration_specifier::Passive::LUT spec,
    const cv::Mat& image,
    cv::Mat& mask)
{
    CV_Assert(!image.empty());
    CV_Assert(image.channels() == 1);
    CV_Assert(m_impl != nullptr);

    cv::Mat image64;
    if (image.type() != CV_64F) image.convertTo(image64, CV_64F);
    else image64 = image;

    // Cause of the limited values
    cv::Mat img_scaled = image64 * 0.9;
    /*double max, min;
    cv::minMaxLoc(img_scaled, &min, &max, nullptr, nullptr);*/

    if (mask.empty()) mask = cv::Mat::ones(image.size(), CV_8U);
    

    return applyLut(img_scaled, mask);
}

cv::Mat GrayCalibration::applyCalibration(
    const GrayCalibration_specifier::Active::Model spec,
    const cv::Mat& image,
    cv::Mat& mask)
{
    // Checks if image is valid
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);

    // Checks if the calibrationdata is available
    CV_Assert(m_impl != nullptr);
    CV_Assert(m_impl->model_a.size() == 7 );
    CV_Assert(std::all_of(m_impl->model_a.begin(), m_impl->model_a.end(),
        [&](const cv::Mat& img) {
            return img.size() == image.size();
        }));

    cv::checkRange(image, false, nullptr, 0.0 - 1e-6, 255.0 + 1e-6);

    const cv::Mat& minImg = m_impl->model_a.at(6);
    const cv::Mat& maxImg = m_impl->model_a.at(5);

    cv::Mat range = maxImg - minImg;
    range /= 255.0;
    range = cv::max(range, 0.0);

    cv::Mat scaled;
    cv::multiply(range, image, scaled);
    scaled += minImg;

    scaled *= 0.8;

    if (mask.empty()) mask = cv::Mat::ones(image.size(), CV_8U);

    cv::Mat cal_img = applyModelFit(scaled, mask, m_impl->model_a);

    boundariesCheck(cal_img, 255.0 + 1e-6, 0 - 1e-6);

    return cal_img;
}

cv::Mat GrayCalibration::applyCalibration(
    const GrayCalibration_specifier::Passive::Model spec,
    const cv::Mat& image,
    cv::Mat& mask)
{
    // Checks if image is valid
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);

    // Checks if the calibrationdata is available
    CV_Assert(m_impl != nullptr);
    CV_Assert(m_impl->model_b.size() > 2); // We should have at least 3 images. 
    CV_Assert(std::all_of(m_impl->model_b.begin(), m_impl->model_b.end(),
        [&](const cv::Mat& img) {
            return img.size() == image.size();
        }));

    if (mask.empty()) mask = cv::Mat::ones(image.size(), CV_8U);

    return applyModelFit(image, mask, m_impl->model_b);
}

cv::Mat GrayCalibration::applyCalibration(
    const GrayCalibration_specifier::Active::Model_Bias spec,
    const cv::Mat& image,
    cv::Mat& mask)
{
    // Checks if image is valid
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);

    // Checks if the calibrationdata is available
    CV_Assert(m_impl != nullptr);
    CV_Assert(m_impl->modelBias_a.size() == 7); // We should have at least 3 images. 
    CV_Assert(std::all_of(m_impl->modelBias_a.begin(), m_impl->modelBias_a.end(),
        [&](const cv::Mat& img) {
            return img.size() == image.size();
        }));

    cv::checkRange(image, false, nullptr, 0.0 - 1e-6, 255.0 + 1e-6);

    const cv::Mat& min = m_impl->modelBias_a.at(6);
    const cv::Mat& max = m_impl->modelBias_a.at(5);

    cv::Mat range = max - min, scaled;

    range /= 255.0;

    range = cv::max(range, 0.0);

    cv::multiply(range, image, scaled);

    scaled += min;

    if (mask.empty()) mask = cv::Mat::ones(image.size(), CV_8U);

    cv::Mat cal_img = applyModelFit(scaled, mask, m_impl->modelBias_a);

    boundariesCheck(cal_img, 255.0 + 1e-6, 0 - 1e-6);

    return cal_img;
}

void GrayCalibration::boundariesCheck(
    cv::Mat& img,
    double maxVal,
    double minVal,
    double threshold)
{
    CV_Assert(!img.empty());
    CV_Assert(img.type() == CV_64F);
    CV_Assert(minVal <= maxVal);
    CV_Assert(threshold >= 0.0);

    img *= 0.9;

    double imageMin, imageMax;
    cv::minMaxLoc(img, &imageMin, &imageMax);

    // too high?
    if (imageMax > maxVal) {
        double overshoot = imageMax - maxVal;

        if (overshoot <= threshold) {
            img -= overshoot;
            cv::minMaxLoc(img, &imageMin, &imageMax);

            if (imageMin < minVal) {
                throw std::runtime_error("Boundary correction pushed image below minimum.");
            }
            return;
        }
        else {
            throw std::runtime_error("Upper boundary exceeded too much.");
        }
    }

    // too low?
    if (imageMin < minVal) {
        double undershoot = minVal - imageMin;

        if (undershoot <= threshold) {
            img += undershoot;
            return;
        }
        else {
            throw std::runtime_error("Lower boundary exceeded too much.");
        }
    }
}
cv::Mat GrayCalibration::applyCalibration(
    const GrayCalibration_specifier::Passive::Model_Bias spec,
    const cv::Mat& image,
    cv::Mat& mask)
{
    // Checks if image is valid
    CV_Assert(!image.empty());

    CV_Assert(image.channels() == 1);

    // Checks if the calibrationdata is available
    CV_Assert(m_impl != nullptr);
    CV_Assert(m_impl->modelBias_b.size() > 2); // We should have at least 3 images. 
    CV_Assert(std::all_of(m_impl->modelBias_b.begin(), m_impl->modelBias_b.end(),
        [&](const cv::Mat& img) {
            return img.size() == image.size();
        }));

    if (mask.empty()) mask = cv::Mat::ones(image.size(), CV_8U);

    cv::Mat img64;
    if (image.type() != CV_64F) image.convertTo(img64, CV_64F);
    else img64 = image;

    return applyModelFit(img64, mask, m_impl->modelBias_b);
}


cv::Mat GrayCalibration::applyModelFit(
    const cv::Mat& image,
    const cv::Mat& mask,
    const std::vector<cv::Mat>& cal_Img)
{
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);

    CV_Assert(mask.type() == CV_8U);
    CV_Assert(mask.channels() == 1);
    CV_Assert(mask.size() == image.size());

    CV_Assert(cal_Img.size() >= 3);
    CV_Assert(std::all_of(cal_Img.begin(), cal_Img.end(),
        [&](const cv::Mat& img) {
            return (img.size() == image.size()) &&
                (img.type() == CV_64F);
        }));

    cv::Mat calibrated(image.size(), CV_64F, cv::Scalar(0));


    cv::parallel_for_(cv::Range(0, image.rows),
        [&](const cv::Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                const uchar* mask_ptr = mask.ptr<uchar>(row);
                const double* gamma_ptr = cal_Img[0].ptr<double>(row);
                const double* i_max_ptr = cal_Img[1].ptr<double>(row);
                const double* I_0_ptr = cal_Img[2].ptr<double>(row);
                const double* img_ptr = image.ptr<double>(row);
                double* cal_ptr = calibrated.ptr<double>(row);
                for (int col = 0; col < image.cols; ++col) {

                    double gamma = gamma_ptr[col];
                    double i_max = i_max_ptr[col];
                    double I0 = I_0_ptr[col];
                    double I = img_ptr[col];

                    if (mask_ptr[col] == 0 ||
                        std::isnan(gamma) ||
                        std::isnan(i_max) ||
                        gamma <= 0.0 ||
                        i_max <= 0.0)
                    {
                        cal_ptr[col] = 0.0;
                        continue;
                    }

                    double normalized = (I-I0) / i_max;
                    normalized = std::max(0.0, normalized);

                    cal_ptr[col] =
                        255.0 * std::pow(normalized, 1.0 / gamma);
                }
            }
        });
    // Clip values to 255 to ensure outliers don't ruin the normalization/display
    /*cv::Mat clipped;
    cv::threshold(calibrated, clipped, 255.0, 255.0, cv::THRESH_TRUNC);

    cv::normalize(clipped, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("img", img);
    cv::waitKey(0);*/

    return calibrated;
}

cv::Mat GrayCalibration::applyLutBackwards(
    const cv::Mat& image,
    const cv::Mat& mask)
{
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(mask.channels() == 1);
    CV_Assert(mask.size() == image.size());

    cv::Mat calibrated(image.size(), CV_64F, cv::Scalar(0));

    cv::parallel_for_(cv::Range(0, image.rows),
        [&](const cv::Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                const double* img_ptr = image.ptr<double>(row);
                double* cal_ptr = calibrated.ptr<double>(row);
                const uchar* mask_ptr = mask.ptr<uchar>(row);
                for (int col = 0; col < image.cols; ++col) {
                    if (mask_ptr[col] == 0 || std::isnan(img_ptr[col])) continue;
                    cal_ptr[col] = getLutvalBackwards(img_ptr[col]);
                }
            }
        });

    return calibrated;
}

cv::Mat GrayCalibration::applyLut(
    const cv::Mat& image,
    const cv::Mat& mask)
{
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(mask.channels() == 1);
    CV_Assert(mask.size() == image.size());

    cv::Mat calibrated(image.size(), CV_64F, cv::Scalar(0));

    cv::parallel_for_(cv::Range(0, image.rows),
        [&](const cv::Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                const double* img_ptr = image.ptr<double>(row);
                double* cal_ptr = calibrated.ptr<double>(row);
                const uchar* mask_ptr = mask.ptr<uchar>(row);
                for (int col = 0; col < image.cols; ++col) {
                    if (mask_ptr[col] == 0 || std::isnan(img_ptr[col])) continue;
                    cal_ptr[col] = getLutVal(img_ptr[col]);
                }
            }
        });

    return calibrated;
}


std::array<std::pair<double, double>, 256> GrayCalibration::createLut(
    const std::vector<cv::Mat>& images,
    cv::Mat& mask) 
{
    CV_Assert(images.size() == 256);

    // Case for valid Mask
    if(!mask.empty()){
        CV_Assert(mask.channels() == 1);
        CV_Assert(std::all_of(images.begin(), images.end(),
            [&](const cv::Mat& img) {
                return (mask.size() == img.size()) &&
                    (img.type() == CV_64F) &&
                    (img.channels() == 1);
            }));
    }
    // Case for empty Mask, New mask gets created with all ones and size of imges[0]
    else {
        CV_Assert(std::all_of(images.begin(), images.end(),
            [&](const cv::Mat& img) {
                return (images[0].size() == img.size()) &&
                    (img.type() == CV_64F) &&
                    (img.channels() == 1);
            }));

        mask = cv::Mat::ones(images[0].size(), CV_8U);
    }
   
    cv::Mat kernel = cv::Mat::ones({ 5,5 }, CV_8U);


    std::array<std::pair<double, double>, 256> Lut{};

    for (std::size_t i = 0; i < images.size(); ++i) {
        double mean = cv::mean(images[i], mask)[0];
        Lut[i].first = static_cast<double>(i);
        Lut[i].second = mean;
    }

    return Lut;
}

double GrayCalibration::getLutvalBackwards(
    const double value)
{
    // Is dataptr set and is in the dataptr the LUT available
    CV_Assert(m_impl != nullptr);
    CV_Assert(!m_impl->empty(GrayCalibration_specifier::Active::LUT{}));

    // binary search on "second" values
    auto it = std::lower_bound(
        m_impl->gray_LUT.gray_Lut.begin(),
        m_impl->gray_LUT.gray_Lut.end(),
        value,
        [](const auto& a, const double val) {
            return a.first < val;
        }
    );

    if (it == m_impl->gray_LUT.gray_Lut.begin())
        return it->second;

    if (it == m_impl->gray_LUT.gray_Lut.end())
        return std::prev(it)->second;

    /*double hi_dist = std::abs(it->first - value);
    double lo_dist = std::abs(std::prev(it)->first - value);

    if (lo_dist < hi_dist) 
        return std::prev(it)->second; 
    
    else return it->second;*/


    auto it_lo = std::prev(it);
    auto it_hi = it;

    const double x0 = it_lo->first;
    const double x1 = it_hi->first;
    const double y0 = it_lo->second;
    const double y1 = it_hi->second;

    if (std::abs(x1 - x0) < 1e-12)
        return y0;

    const double t = (value - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}


double GrayCalibration::getLutVal(
    const double value)
{
    // Is dataptr set and is in the dataptr the LUT available
    CV_Assert(m_impl != nullptr);
    CV_Assert(!m_impl->empty(GrayCalibration_specifier::Active::LUT{}));

    // binary search on "second" values
    auto it = std::lower_bound(
        m_impl->gray_LUT.gray_Lut.begin(),
        m_impl->gray_LUT.gray_Lut.end(),
        value,
        [](const auto& a, const double val) {
            return a.second < val;
        }
    );

    if (it == m_impl->gray_LUT.gray_Lut.begin())
        return it->first;

    if (it == m_impl->gray_LUT.gray_Lut.end())
        return std::prev(it)->first;

    auto it_lo = std::prev(it);
    auto it_hi = it;

    const double x0 = it_lo->second;
    const double x1 = it_hi->second;
    const double y0 = it_lo->first;
    const double y1 = it_hi->first;

    // Schutz gegen degenerierten Fall
    if (std::abs(x1 - x0) < 1e-12)
        return y0;

    const double t = (value - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}


Gray_Calib_Result GrayCalibration::fitGammaBias_LM(
    const std::array<double, 256>& meassured,
    const double eps,
    const double sat_cut)
{
    CV_Assert(!meassured.empty());
    CV_Assert(eps >= 0 && sat_cut > 0);

    double i_min{ 0 }, gamma{ 1.0 }, i_max{ 150.0 };

    Gray_Calib_Result result;

    std::vector<detail::Sample> values =
        detail::buildSamples(meassured, sat_cut, eps);

    if (values.empty()) {
        result.stats.gamma = result.stats.Imax = result.stats.I_0 =
            result.stats.r2 = result.stats.rmse = std::numeric_limits<double>::quiet_NaN();

        result.stats.iters = result.stats.n = 0;

        result.stats.converged = false;

        return result;
    }

    ceres::Problem problem;

    for (std::size_t i = 0; i < values.size(); ++i) {
        auto* costfunction =
            new ceres::AutoDiffCostFunction<ceresCost::ExponentialResidual,
            1, 1, 1, 1>(new ceresCost::ExponentialResidual(values[i].u, values[i].I));

        problem.AddResidualBlock(costfunction, nullptr, &i_max, &gamma, &i_min);

        problem.SetParameterLowerBound(&gamma, 0, 0.01);
        problem.SetParameterLowerBound(&i_max, 0, 1e-6);
        problem.SetParameterLowerBound(&i_min, 0, 0.0);

        problem.SetParameterUpperBound(&gamma, 0, 8.0);
        problem.SetParameterUpperBound(&i_max, 0, 255.0);
        problem.SetParameterUpperBound(&i_min, 0, 255.0);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   
    //options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.stats.gamma = gamma;
    result.stats.Imax = i_max;
    result.stats.I_0 = i_min;
    
    result.stats.r2 = detail::computeR2(values, i_max, gamma, eps, i_min);

    return result;
}

Gray_Calib_Result GrayCalibration::fitGamma(
    const std::array<double, 256>& meassured,
    const double eps,
    const double sat_cut)
{
    std::vector<detail::Sample> samples =
        detail::buildSamples(meassured, sat_cut, eps);
    
    Gray_Calib_Stats stats;

    if (samples.empty()) {
        stats.gamma =  stats.Imax = stats.I_0 = 
           stats.r2 = stats.rmse = std::numeric_limits<double>::quiet_NaN();

        stats.iters = stats.n = 0;

        stats.converged = false;

        Gray_Calib_Result result;

        result.stats = stats;

        return result;
    }

    double ln_x{}, ln_y{}, ln_x2{}, ln_y2{}, ln_xy{};

    for (std::size_t val = 0; val < samples.size(); ++val) {
        CV_Assert(samples[val].I > 0);
        CV_Assert(samples[val].u > 0);

        const double ln_x_act = std::log(samples[val].u);
        ln_x += ln_x_act;
        ln_x2 += ln_x_act * ln_x_act;
        const double ln_y_act = std::log(samples[val].I);
        ln_y += ln_y_act;
        ln_y2 += ln_y_act * ln_y_act;
        ln_xy += ln_x_act * ln_y_act;
    }
    
    double n = static_cast<double>(samples.size());
    
    double gamma_num =
        ln_xy - (ln_x * ln_y) / n;

    double gamma_denom =
        - (ln_x/n) * (ln_x) + ln_x2;

    double gamma = gamma_num / gamma_denom;

    double i_max = std::exp((ln_y - gamma * ln_x) / n);

    double sse = detail::computeSSE(samples, i_max, gamma);

    double r_2 = detail::computeR2(samples, i_max, gamma, eps);

    stats.gamma = gamma;
    stats.Imax = i_max;
    stats.n = static_cast<int>(n);
    stats.r2 = r_2;
    stats.sse = sse;
    stats.rmse = std::sqrt(sse / std::max(1, stats.n - 2)); // dof n-2

    Gray_Calib_Result result;

    result.stats = stats;

    return result;
}

GrayCalibration::GrayCalibration(ImageStore& imgStore)
	:m_img_store{ imgStore }
	,m_impl{ std::make_unique<Impl>() }
	{}

GrayCalibration::~GrayCalibration() = default;


bool GrayCalibration::setupCalibrationMethod(
	const GrayCalibration_specifier::Active::LUT spec,
	const std::string& path)
{
    CV_Assert(!path.empty());
    
    if (!m_impl->empty(spec)) {
        std::cout << "WARNING --- DataPointer does already contain data \n" <<
            "Stop loading LUT and return \n";
        return false;
    }

    m_img_store.loadRoleXML(FrameRole::GrayLUT, path);
    m_impl->gray_LUT.gray_Lut = m_img_store.getLut();

    prepareLUT();

    // If after loading still empty return false. Should produce hard error before.
    if (m_impl->empty(spec)) return false;
    
    return true;
}

bool GrayCalibration::setupCalibrationMethod(
    const GrayCalibration_specifier::Passive::LUT,
    const std::string& path)
{
    return setupCalibrationMethod(
        GrayCalibration_specifier::Active::LUT{},
        path);
}

bool GrayCalibration::setupCalibrationMethod(
    const GrayCalibration_specifier::Active::Model spec,
    const std::string& path)
{
    CV_Assert(!path.empty());

    if (!m_impl->empty(spec)) {
        std::cout << "WARNING --- DataPointer does already contain data \n" <<
            "Stop and return \n";
        return false;
    }

    m_img_store.loadRoleXML(FrameRole::Modell_Active, path);
    m_impl->model_a = m_img_store.get(FrameRole::Modell_Active);

    smoothModelImages(m_impl->model_a);

    if (m_impl->empty(spec)) return false;

    return true;
}

bool GrayCalibration::setupCalibrationMethod(
    const GrayCalibration_specifier::Active::Model_Bias spec,
    const std::string& path)
{
    CV_Assert(!path.empty());

    if (!m_impl->empty(spec)) {
        std::cout << "WARNING --- DataPointer does already contain data \n" <<
            "Stop and return \n";
        return false;
    }

    m_img_store.loadRoleXML(FrameRole::ModellBias_Active, path);
    m_impl->modelBias_a = m_img_store.get(FrameRole::ModellBias_Active);

    smoothModelImages(m_impl->modelBias_a);

    if (m_impl->empty(spec)) return false;

    return true;
}

bool GrayCalibration::setupCalibrationMethod(
	const GrayCalibration_specifier::Passive::Model spec,
	const std::string& path)
{
    CV_Assert(!path.empty());

    if (!m_impl->empty(spec)) {
        std::cout << "WARNING --- DataPointer does already contain data \n" <<
            "Stop and return \n";
        return false;
    }

    m_img_store.loadRoleXML(FrameRole::Modell_Passive, path);
    m_impl->model_b = m_img_store.get(FrameRole::Modell_Passive);

    smoothModelImages(m_impl->model_b);

    if (m_impl->empty(spec)) return false;

    return true;
}

bool GrayCalibration::setupCalibrationMethod(
	const GrayCalibration_specifier::Passive::Model_Bias spec,
    const std::string& path)
{
    CV_Assert(!path.empty());

    if (!m_impl->empty(spec)) {
        std::cout << "WARNING --- DataPointer does already contain data \n" <<
            "Stop and return \n";
        return false;
    }

    m_img_store.loadRoleXML(FrameRole::ModellBias_Passive, path);
    m_impl->modelBias_b = m_img_store.get(FrameRole::ModellBias_Passive);

    smoothModelImages(m_impl->modelBias_b);

    if (m_impl->empty(spec)) return false;

    return true;
}

bool GrayCalibration::prepareLUT()
{
    std::cout << "Prepare LUT \n";

    double safety = 0.95;

    // Small valid Check is max val != 0 
    // Also check for Lut values != 0 if there are values Lut must be filled
    CV_Assert(m_impl->gray_LUT.m_max_LUT == 0);
    CV_Assert(std::any_of(m_impl->gray_LUT.gray_Lut.begin(),
        m_impl->gray_LUT.gray_Lut.end(),
        [&](const std::pair<double, double>& val) {
            return (val.first != 0) || (val.second != 0);
        }));

    try {
        std::sort(m_impl->gray_LUT.gray_Lut.begin(), m_impl->gray_LUT.gray_Lut.end(),
            [](auto& a, auto& b) { return a.second < b.second; });

        double minv = m_impl->gray_LUT.gray_Lut.front().second;
        double maxv = m_impl->gray_LUT.gray_Lut.back().second;

        m_impl->gray_LUT.m_min_LUT = minv;
        m_impl->gray_LUT.m_max_LUT = maxv;

        double range = maxv - minv;
        double range_safety = range * safety;      // keep 10% margin
                
        double scale_factor = m_impl->gray_LUT.m_scale_factorLUT
            = range_safety / 255.0;

        // These bias values and scaling must be applied at the active case. 

        double bias = (1-safety)/1.25 * 255.0; 

        m_impl->gray_LUT.m_biasLut = bias;
    }
    catch (std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return false;
    }
    return true;
}
