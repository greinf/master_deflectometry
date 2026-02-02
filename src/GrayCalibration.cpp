#include "GrayCalibration.hpp"
#include "cassert"
#include <memory>
#include "imageStore.hpp"
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>
#include "utils.hpp"
#include <ceres/ceres.h>

struct GrayCalibration::Impl {
	std::vector<cv::Mat> active_gray{};
	std::vector<cv::Mat> passive_gray{};
	std::array<std::pair<double, double>, 256> gray_Lut{};
};



// ----------   Passive Camera Calibration with Model  ------------------
struct Gray_Calib_Stats {
    int n = 0;
    int iters = 0;
    bool converged = false;

    double gamma = 1.0;
    double Imax = 1.0;
    double I_0 = 0.0;

    double sse = 0.0;
    double rmse = 0.0;
    double r2 = 0.0;
};

struct Gray_Calib_Result {
    Gray_Calib_Stats stats;
    std::array<uint8_t, 256> lut{};
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
        double u; // g/255
        double I; // measured intensity
    };


    // Try to do some data sampling before we start 
    // 
    inline std::vector<Sample> buildSamples(const std::array<double, 256>& measured,
        double sat_cut_rel,
        double eps)
    {
        // If the biggest Element is smaller than epsion than Abort directly
        double Imax_obs = *std::max_element(measured.begin(), measured.end());
        if (Imax_obs < eps) throw std::runtime_error("No valid signal (Imax_obs ~ 0).");

        std::vector<Sample> samples;
        samples.reserve(256);

        for (int g = 0; g <= 255; ++g) {
            double I = measured[g];
            // We cut first if the smaller than epsion.
            if (I < eps) continue;

             
            if ((I / Imax_obs) > sat_cut_rel) {
                //std::cout << "WARNING: Saturated Values were cut in Calibration \n";
                continue;
            }

            // normalize to value 0 ... 1
            double u = static_cast<double>(g) / 255.0;
            if (u <= 0.0) continue;

            // Save the intensity Val 0 ... 255 I and normalized to 0 ... 1
            samples.push_back({ u, I });
        }
        if (samples.size() < 10) throw std::runtime_error("Not enough samples for LM fit.");
        return samples;
    }

    inline double predict(double Imax, double gamma, double u) {
        return Imax * std::pow(u, gamma);
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
    inline double computeR2(const std::vector<Sample>& s, double Imax, double gamma, double eps) {
        double meanI = 0.0;
        for (auto& p : s) meanI += p.I;
        meanI /= std::max<size_t>(1, s.size());

        double sst = 0.0, sse = 0.0;
        for (auto& p : s) {
            double yhat = predict(Imax, gamma, p.u);
            double e = p.I - yhat;
            sse += e * e;

            double d = p.I - meanI;
            sst += d * d;
        }
        if (!(sst > eps)) return 0.0;
        return 1.0 - sse / sst;
    }

} // namespace detail

cv::Mat GrayCalibration::applyCalibration(
    const CalibrationMethod method,
    const cv::Mat& image) 
{
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);

    switch (method) {
    case(CalibrationMethod::None):
        std::cout << "No Calibration Used " << std::endl;
        return image;
    case(CalibrationMethod::Lut):
        return applyLut(image);
    case(CalibrationMethod::Passive):
        [[fallthrough]];
    case(CalibrationMethod::Bias_Passive):
        return applyPassive(image);
    case(CalibrationMethod::Active):
        std::cout << "not Calibrated " << std::endl;
        return image;
    }
}

cv::Mat GrayCalibration::applyPassive(
    const cv::Mat& image)
{
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);

    CV_Assert(m_impl->passive_gray.size() == 3);
    CV_Assert(std::all_of(m_impl->passive_gray.begin(), m_impl->passive_gray.end(),
        [&](const cv::Mat& img) {
            return img.size() == image.size();
        }));

    cv::Mat calibrated(image.size(), CV_64F, cv::Scalar(0));

    cv::parallel_for_(cv::Range(0, image.rows),
        [&](const cv::Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                const double* gamma_ptr = m_impl->passive_gray[0].ptr<double>(row);
                const double* i_max_ptr = m_impl->passive_gray[1].ptr<double>(row);
                const double* I_0_ptr = m_impl->passive_gray[2].ptr<double>(row);
                const double* img_ptr = image.ptr<double>(row);
                double* cal_ptr = calibrated.ptr<double>(row);
                for (int col = 0; col < image.cols; ++col) {
                    cal_ptr[col] =
                        I_0_ptr[col] + std::pow(img_ptr[col] / i_max_ptr[col], 1.0 / gamma_ptr[col]) * 255.0;
                }
            }
        });

    return calibrated;
}


cv::Mat GrayCalibration::applyLut(
    const cv::Mat& image)
{
    CV_Assert(image.type() == CV_64F);
    CV_Assert(image.channels() == 1);
    
    cv::Mat calibrated(image.size(), CV_64F, cv::Scalar(0));

    cv::parallel_for_(cv::Range(0, image.rows),
        [&](const cv::Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                const double* img_ptr = image.ptr<double>(row);
                double* cal_ptr = calibrated.ptr<double>(row);
                for (int col = 0; col < image.cols; ++col) {
                    cal_ptr[col] = getLutVal(img_ptr[col]);
                }
            }
        });

    return calibrated;
}



Gray_Calib_Result GrayCalibration::fitGamma_LM_andBuildLUT(
    const std::array<double, 256>& measured,
    double sat_cut_rel,    // Sättigung raus
    double eps,
    int max_iters,
    double tol_rel_sse,
    double tol_step,
    double damping)
{
    using namespace detail;

    std::vector<detail::Sample> samples = buildSamples(measured, sat_cut_rel, eps);

    // --- Initial Guess ---
    double Imax0{};
    // Iterate through samles and take biggest I_max
    for (auto& p : samples) Imax0 = std::max(Imax0, p.I);  

    // Another validity check if I max is really bigger than eps
    double Imax{};
    if (Imax0 > eps) Imax = Imax0;
    else throw std::runtime_error("I max is not allwed to be smaller than eps \n");
   
    // gamma: grob aus zwei Punkten schätzen (robust: mittlere u)
    // gamma ~ log(I/Imax)/log(u)
    double gamma = 1.0;
    {
        // pick a sample from the middle ca 
        auto it = std::min_element(samples.begin(), samples.end(),
            [](const Sample& a, const Sample& b) {
                return std::abs(a.u - 0.5) < std::abs(b.u - 0.5);
            });

        // Try to calc for a middle point smaple for one time the gamma as initial starting value 
        if (it != samples.end()) {
            double u = it->u;
            double I = it->I;
            if (I > eps && u > 0.0 && u < 1.0) {
                double g = std::log(std::max(I / Imax, eps)) / std::log(u);
                // we have to test if g is finit and > 0. if not take gamma 1.0
                if (isFinite(g) && g > 0.0) gamma = clamp(g, 0.1, 10.0);
            }
        }
    }

    // Initiale SSE
    double sse = computeSSE(samples, Imax, gamma);
    if (!isFinite(sse)) throw std::runtime_error("Initial SSE not finite.");

    Gray_Calib_Stats stats;
    stats.n = static_cast<int>(samples.size());

    bool converged = false;
    for (int it = 0; it < max_iters; ++it) {
        stats.iters = it + 1;

        // Build normal equations: (J^T J) dp = J^T r
        // Parameters p = [Imax, gamma]
        double A00 = 0.0, A01 = 0.0, A11 = 0.0;
        double b0 = 0.0, b1 = 0.0;

        for (const auto& p : samples) {
            const double u = p.u;
            const double I = p.I;

            // model
            const double ug = std::pow(u, gamma);

            // yhat approximation of value 
            const double yhat = Imax * ug;

            // I real meassruement. 
            const double r = I - yhat;

            // Jacobian of yhat:
            // dy/dImax = u^gamma
            // dy/dgamma = Imax * u^gamma * ln(u)
            const double J0 = ug;
            const double J1 = Imax * ug * std::log(u);

            // J^T J
            A00 += J0 * J0;
            A01 += J0 * J1;
            A11 += J1 * J1;

            // J^T r
            b0 += J0 * r;
            b1 += J1 * r;
        }


        // Solve 2x2 system:
        // [A00 A01][dImax ] = [b0]
        // [A01 A11][dgamma] = [b1]
        const double det = A00 * A11 - A01 * A01;
        if (!(std::abs(det) > eps) || !isFinite(det)) {
            std::cout << "WARNING: det -> 0";
            continue;
        }

        const double dImax = (b0 * A11 - b1 * A01) * damping / det;
        const double dgamma = (-b0 * A01 + b1 * A00) *damping  / det;

        if (!isFinite(dImax) || !isFinite(dgamma)) {
            std::cout << "WARNING: Values are not finite ";;
            continue;
        }

        /*std::cout << "Old IMax " << Imax << '\n';
        std::cout << "Update Value  + " << dImax << '\n';

        std::cout << "old Gamma " << gamma << '\n';
        std::cout << "update Gamma " << dgamma << '\n';*/

        // update the Imax & gamma
        double Imax_trial = Imax + dImax;
        double gamma_trial = gamma + dgamma;

        // validtiy check
        Imax_trial = std::max(Imax_trial, eps);
        gamma_trial = std::max(gamma_trial, eps);

        double sse_trial = computeSSE(samples, Imax_trial, gamma_trial);

        

        if (!isFinite(sse_trial)) {
            std::cout << "SSE value is not finite;";
            continue;
        }

        /*std::cout << "new sse: " << sse_trial << "\n";
        std::cout << "old sse: " << sse << '\n';*/

        // Accept / reject
        if (sse_trial < sse) {
            // accept
            const double rel_impr = (sse - sse_trial) / std::max(sse, eps);
            Imax = Imax_trial;
            gamma = gamma_trial;
            sse = sse_trial;

            // convergence checks
            const double step_norm = std::sqrt(dImax * dImax + dgamma * dgamma);
            if (rel_impr < tol_rel_sse || step_norm < tol_step) {
                converged = true;
                break;
            }
        }
        else {
            
            std::cout << "Error is not decreasing \n";
            
        }
    }

    // final stats
    stats.converged = converged;
    stats.Imax = Imax;
    stats.gamma = gamma;
    stats.sse = sse;
    stats.rmse = std::sqrt(sse / std::max(1, stats.n - 2)); // dof n-2
    stats.r2 = computeR2(samples, Imax, gamma, eps);

    // LUT (inverse gamma) für Linearisierung der Anzeige:
    // target linear i/255 -> corrected input = 255*(i/255)^(1/gamma)
    std::array<uint8_t, 256> lut{};
    lut[0] = 0;
    for (int i = 1; i <= 255; ++i) {
        double p = static_cast<double>(i) / 255.0;
        double gc = 255.0 * std::pow(p, 1.0 / gamma);
        int gi = static_cast<int>(std::lround(gc));
        gi = std::clamp(gi, 0, 255);
        lut[i] = static_cast<uint8_t>(gi);
    }

    Gray_Calib_Result out;
    out.stats = stats;
    out.lut = lut;
    return out;
}

std::array<std::pair<double, double>, 256> GrayCalibration::createLut(
    const std::vector<cv::Mat>& images,
    const cv::Mat& mask) 
{
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(mask.channels() == 1);
    CV_Assert(images.size() == 256);
    CV_Assert(std::all_of(images.begin(), images.end(),
        [&](const cv::Mat& img) {
            return mask.size() == img.size() && 
                img.type() == CV_64F && 
                img.channels() == 1;
        }));

    std::array<std::pair<double, double>, 256> Lut{};

    for (std::size_t i = 0; i < images.size(); ++i) {
        double mean = cv::mean(images[i], mask)[0];
        Lut[i].first = static_cast<double>(i);
        Lut[i].second = mean;
    }

    return Lut;
}

double GrayCalibration::getLutVal(
    const double value)
{
    CV_Assert(!m_impl->gray_Lut.empty());
    CV_Assert(m_impl->gray_Lut.begin() != 
        m_impl->gray_Lut.end());

    // binary search on "second" values
    auto it = std::lower_bound(
        m_impl->gray_Lut.begin(),
        m_impl->gray_Lut.end(),
        value,
        [](const auto& a, double val) {
            return a.second < val;
        }
    );

    if (it == m_impl->gray_Lut.begin())
        return it->first;

    if (it == m_impl->gray_Lut.end())
        return std::prev(it)->first;

    // choose closer of the two neighbors
    double hi_dist = std::abs(it->second - value);
    double lo_dist = std::abs(std::prev(it)->second - value);

    if (lo_dist < hi_dist)
        return std::prev(it)->first;
    else
        return it->first;
}


bool GrayCalibration::doCalibration(
    const CalibrationMethod method,
    const std::vector<cv::Mat>& images,
    const cv::Mat& mask)
{
    CV_Assert(!images.empty());
    CV_Assert(!mask.empty());
    CV_Assert(images.size() == 256);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(m_impl != nullptr);

    for (const auto& im : images) {
        CV_Assert(im.size() == mask.size());
        CV_Assert(im.type() == CV_64FC1);  
    }

    if (method == CalibrationMethod::Lut) {
        std::array<std::pair<double,double>, 256> lut =
            createLut(images, mask);
        
        if (!prepareLUT(lut)) {
            std::cout << "Lut Creation failed \n";
            return false;
        }

        m_impl->gray_Lut = lut;

        m_img_store.add(FrameRole::GrayLUT, lut);

        return true;
    }

    cv::Mat gamma(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat Imax(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat I_0(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat errorR2(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat iter(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));


    cv::parallel_for_(cv::Range(0, mask.rows),
        [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; ++r) {
                const uchar* maskPtr = mask.ptr<uchar>(r);

                double* gammaPtr = gamma.ptr<double>(r);
                double* ImaxPtr = Imax.ptr<double>(r);
                double* I_0Ptr = I_0.ptr<double>(r);
                double* r2Ptr = errorR2.ptr<double>(r);
                double* iterPtr = iter.ptr<double>(r);

                for (int c = 0; c < mask.cols; ++c) {
                    if (maskPtr[c] == 0) continue;

                    std::array<double, 256> y{};
                    for (int i = 0; i < 256; ++i) {
                        y[i] = images[i].ptr<double>(r)[c];
                    }

                    try {
                        switch (method) {
                        case(CalibrationMethod::Active):
                            std::cout << "Active Calibration is not implemented so no \n";
                            return false;
                            break;
                        case(CalibrationMethod::Bias_Passive): {
                            auto res = fitGammaBias_LM(y);
                            gammaPtr[c] = res.stats.gamma;
                            ImaxPtr[c] = res.stats.Imax;
                            I_0Ptr[c] = res.stats.I_0;
                            r2Ptr[c] = res.stats.r2;
                            iterPtr[c] = res.stats.iters;
                            break;
                        }
                        case(CalibrationMethod::Passive): {
                            auto res = fitGamma(y);
                            gammaPtr[c] = res.stats.gamma;
                            ImaxPtr[c] = res.stats.Imax;
                            I_0Ptr[c] = res.stats.I_0;
                            r2Ptr[c] = res.stats.r2;
                            iterPtr[c] = res.stats.iters;
                            break;
                        }
                        
                        }
                    }
                    catch (const std::exception&) {
                        
                    }
                }
            }
        });

    // Both Passive and Bias_Passive have essnetially the same procedure apart from the creation. 
    if (method == CalibrationMethod::Passive || 
        method == CalibrationMethod::Bias_Passive) {
        m_img_store.add(FrameRole::PassiveGrayCalib, gamma);
        m_img_store.add(FrameRole::PassiveGrayCalib, Imax);
        m_img_store.add(FrameRole::PassiveGrayCalib, I_0);
        m_img_store.add(FrameRole::PassiveGrayCalib, errorR2);
        m_img_store.add(FrameRole::PassiveGrayCalib, iter);
    }

    return true;
}

Gray_Calib_Result GrayCalibration::fitGammaBias_LM(
    const std::array<double, 256>& meassured,
    const double eps,
    const double sat_cut)
{
    CV_Assert(!meassured.empty());
    CV_Assert(eps >= 0 && sat_cut > 0);

    double i_min{ 0 }, gamma{ 1.0 }, i_max{ 150.0 };

    std::vector<detail::Sample> values =
        detail::buildSamples(meassured, sat_cut, eps);

    ceres::Problem problem;

    for (std::size_t i = 0; i < values.size(); ++i) {
        auto* costfunction =
            new ceres::AutoDiffCostFunction<ceresCost::ExponentialResidual,
            1, 1, 1, 1>(new ceresCost::ExponentialResidual(values[i].u, values[i].I));

        problem.AddResidualBlock(costfunction, nullptr, &i_max, &gamma, &i_min);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   
    //options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //std::cout << summary.BriefReport() << "\n";


    Gray_Calib_Result result;
    result.stats.gamma = gamma;
    result.stats.Imax = i_max;
    result.stats.I_0 = i_min;
    result.stats.r2 = summary.final_cost;

    return result;

}



Gray_Calib_Result GrayCalibration::fitGamma(
    const std::array<double, 256>& meassured,
    const double eps,
    const double sat_cut)
{
    std::vector<detail::Sample> samples =
        detail::buildSamples(meassured, sat_cut, eps);
    
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
    
    Gray_Calib_Stats stats;

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
    stats.n = n;
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
	const gr_calib::ActiveCalibration,
	const std::string& path)
{
    CV_Assert(!path.empty());

    m_img_store.loadRoleXML(FrameRole::AcitveGrayCalib, path);

    m_impl->active_gray = m_img_store.get(FrameRole::AcitveGrayCalib);
	
    if (!m_impl->active_gray.size() == 2) return false;

	return true;
}

bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::PassiveCalibration,
	const std::string& path)
{
    CV_Assert(!path.empty());

    m_img_store.loadRoleXML(FrameRole::PassiveGrayCalib, path);

    // Frame Role can have up to 5 Pictures stored. Only the first three are needed for the calibration
    std::vector<cv::Mat> passive_calib = 
        m_img_store.get(FrameRole::PassiveGrayCalib);

    m_impl->passive_gray =
        std::vector<cv::Mat>(passive_calib.begin(), std::next(passive_calib.begin(), 3));

    if (!m_impl->passive_gray.size() == 3) return false;

    return true;
}

bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::LUTCalibration,
    const std::string& path)
{
    CV_Assert(!path.empty());
    if(!m_img_store.has(FrameRole::GrayLUT))
        m_img_store.loadLut(path);

    m_impl->gray_Lut = m_img_store.getLut();

    if (!m_impl->gray_Lut.size() == 256) return false;

    return true;
}


bool GrayCalibration::prepareLUT(
    std::array<std::pair<double,double>,256>& lut)
{
    CV_Assert(!lut.empty());
    try {
        std::sort(lut.begin(), lut.end(),
            [](auto& a, auto& b) { return a.second < b.second; });

        double minv = lut.front().second;
        double maxv = lut.back().second;

        double range = maxv - minv;
        double range_safety = range * 0.9;      // keep 10% margin
    }
    catch (std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return false;
    }


    return true;
}
