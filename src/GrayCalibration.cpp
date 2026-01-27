#include "GrayCalibration.hpp"
#include "cassert"
#include <memory>
#include "imageStore.hpp"
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>
#include "utils.hpp"


struct GrayCalibration::Impl {
	std::vector<cv::Mat> active_gray{};
	std::vector<cv::Mat> passive_gray{};
	std::vector<std::pair<double, double>> gray_Lut{};
};



// ----------   Passive Camera Calibration with Model  ------------------
struct GammaLMStats {
    int n = 0;
    int iters = 0;
    bool converged = false;

    double gamma = 1.0;
    double Imax = 1.0;

    double sse = 0.0;
    double rmse = 0.0;
    double r2 = 0.0;
};

struct GammaLMResult {
    GammaLMStats stats;
    std::array<uint8_t, 256> lut{};
};

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
    inline std::vector<Sample> buildSamples(const std::array<double, 256>& measured,
        double sat_cut_rel,
        double eps)
    {
        // If the biggest Element is smaller than epsion than Abort directly
        double Imax_obs = *std::max_element(measured.begin(), measured.end());
        if (Imax_obs < eps) throw std::runtime_error("No valid signal (Imax_obs ~ 0).");

        std::vector<Sample> samples;
        samples.reserve(256);

        for (int g = 1; g <= 255; ++g) {
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




GammaLMResult GrayCalibration::fitGamma_LM_andBuildLUT(
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

    GammaLMStats stats;
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

    GammaLMResult out;
    out.stats = stats;
    out.lut = lut;
    return out;
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

    for (const auto& im : images) {
        CV_Assert(im.size() == mask.size());
        CV_Assert(im.type() == CV_64FC1);  
    }

    cv::Mat gamma(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat Imax(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat errorR2(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
    cv::Mat iter(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));

    cv::parallel_for_(cv::Range(0, mask.rows),
        [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; ++r) {
                const uchar* maskPtr = mask.ptr<uchar>(r);

                double* gammaPtr = gamma.ptr<double>(r);
                double* ImaxPtr = Imax.ptr<double>(r);
                double* r2Ptr = errorR2.ptr<double>(r);
                double* iterPtr = iter.ptr<double>(r);

                for (int c = 0; c < mask.cols; ++c) {
                    if (maskPtr[c] == 0) continue;

                    std::array<double, 256> y{};
                    for (int i = 0; i < 256; ++i) {
                        y[i] = images[i].ptr<double>(r)[c];
                    }

                    try {
                        if (method == CalibrationMethod::Passive) {
                            auto res = fitGamma_LM_andBuildLUT(y, 0.9995);

                            gammaPtr[c] = res.stats.gamma;
                            ImaxPtr[c] = res.stats.Imax;
                            r2Ptr[c] = res.stats.r2;
                            iterPtr[c] = res.stats.iters;
                            
                        }
                    }
                    catch (const std::exception&) {
                        
                    }
                }
            }
        });

    if (method == CalibrationMethod::Passive) {
        m_img_store.add(FrameRole::PassiveGrayCalib, gamma);
        m_img_store.add(FrameRole::PassiveGrayCalib, Imax);
        m_img_store.add(FrameRole::PassiveGrayCalib, errorR2);
        m_img_store.add(FrameRole::PassiveGrayCalib, iter);
    }

    return true;
}


GrayCalibration::GrayCalibration(ImageStore& imgStore)
	:m_img_store{ imgStore }
	,m_impl{ std::make_unique<Impl>() }
	{}

GrayCalibration::~GrayCalibration() = default;



bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::ActiveCalibration,
	const std::string& pat)
{
	
	

	return false;
}

bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::PassiveCalibration,
	const std::string& path)
{
	assert(!path.empty() && "Path to active Calibration Frames is needed \n");
	return false;
}

bool GrayCalibration::setupCalibrationMethod(
	const gr_calib::LUTCalibration,
	std::vector<std::pair<double,double>> grayLut)
{
	assert(!grayLut.empty());
	assert(grayLut.size() > 255);
	m_impl->gray_Lut = grayLut;


	return true;
}


//void GrayCalibration::prepareLUT()
//{
//	if (!m_LUT.has_value()) {
//		m_lut_ready = false;
//		return;
//	}
//
//	m_sortedLUT = m_LUT.value();
//	std::sort(m_sortedLUT.begin(), m_sortedLUT.end(),
//		[](auto& a, auto& b) { return a.second < b.second; });
//
//	double minv = m_sortedLUT.front().second;
//	double maxv = m_sortedLUT.back().second;
//
//	double range = maxv - minv;
//	double range_safety = range * 0.9;      // keep 10% margin
//
//	m_lut_scale_factor = range_safety / 255.0;
//	m_lut_offset = minv + range * 0.05;
//
//	m_lut_ready = true;
//}
