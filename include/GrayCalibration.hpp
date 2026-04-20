#ifndef GRAYCALIBRATION_HPP
#define GRAYCALIBRATION_HPP

#include <vector>
#include <type_traits>
#include <array>
#include <string>
#include <algorithm>
#include <memory>
#include "RowPolicy.hpp"
#include <opencv2/core.hpp>
#include "GrayCalibration_Utils.hpp"

class ImageStore;


class GrayCalibration {
private:
	
	ImageStore& m_img_store;

	struct Impl;        

	std::unique_ptr<Impl> m_impl = nullptr;  
	
	// Fits a function of type I_mess = I_max * (x/255)^lambda 
	// for  the parameters lambda and I_max. This only works if there is no bias 
	// Parameter eps is used for cutting values nere zero -> val < eps is discarded
	// paramter sat_cut cuts values that are near the 
	Gray_Calib_Result fitGamma(
		const std::array<double, 256>& meassured,
		const double eps = 1e-10,
		const double sat_cut = 1
	);

    // Method for creating a LUT. Can be used for both active and passive
    // Mask can be empy. If empty all Pixels processed as valid
	std::array<std::pair<double, double>, 256> createLut(
		const std::vector<cv::Mat>& images,
		cv::Mat& mask
	);

	// Sorts the Lut Array in the impl ptr. 
	bool prepareLUT();

	cv::Mat applyLut(
		const cv::Mat& img,
        const cv::Mat& mask
	);

	double getLutVal(
		const double val
	);

	cv::Mat applyModelFit(
		const cv::Mat& img,
        const cv::Mat& mask,
        const std::vector<cv::Mat>& calImages
	);

	Gray_Calib_Result fitGammaBias_LM(
		const std::array<double, 256>& meassured,
		const double eps = 1e-10,
		const double sat_cut = 1
	);


    // Transporter methods to connect the hpp method to impl 
    void updateLut(const std::array<std::pair<double, double>, 256> LUT);

    void updateModelActive(const std::vector<cv::Mat>);

    void updateModelPassive(const std::vector<cv::Mat>);
    
    void updateModel_BiasActive(const std::vector<cv::Mat>);

    void updateModel_BiasPassive(const std::vector<cv::Mat>);

    void smoothModelImages(std::vector<cv::Mat>&, int kernel = 15);

    void boundariesCheck(cv::Mat&, double max, double min, double threshold = 3);

    cv::Mat applyLutBackwards(const cv::Mat& img, const cv::Mat& mask);

    double getLutvalBackwards(const double);

public:
    // GrayCalibration Must have a member reference to the ImageStore instance (internally 
	explicit GrayCalibration(ImageStore& image_store);
	~GrayCalibration();

    // ***** Setup *****
    // Input[0]: Specifier for used method. (-> Struct for this in GrayCalibrationUtils.hpp)
    // Input[1]: const std::string& path 
    // (->Path to the .xml files for Model or Model Bias - Or .csv for LUT)
    // Output: bool value if loading data was succesfull
    
    // Setup the data for Calibration Method (Active-LUT) in Impl-ptr
    // Active LUT and passive LUT are done the same way 
	bool setupCalibrationMethod(
		const GrayCalibration_specifier::Active::LUT,
		const std::string& path);

    // Setup the data for Calibration Method (Active-Model) in Impl-ptr
	bool setupCalibrationMethod(
		const GrayCalibration_specifier::Active::Model,
		const std::string& path);

    // Setup the data for Calibration Method (Active-ModelBias) in Impl-ptr
	bool setupCalibrationMethod(
		const GrayCalibration_specifier::Active::Model_Bias,
		const std::string& path);

    // Setup the data for Calibration Method (Passive-LUT) in Impl-ptr
    // Just calls the Active::LUT method
	bool setupCalibrationMethod(
		const GrayCalibration_specifier::Passive::LUT,
		const std::string& path);

    // Setup the data for Calibration Method (Passive-Modell) in Impl-ptr
	bool setupCalibrationMethod(
		const GrayCalibration_specifier::Passive::Model,
		const std::string& path);

    // Setup the data for Calibration Method (Passive-ModellBias) in Impl-ptr
	bool setupCalibrationMethod(
		const GrayCalibration_specifier::Passive::Model_Bias,
		const std::string& path);

    // Setup the data for Calibration Method (No Calib)
    bool setupCalibrationMethod(
        const GrayCalibration_specifier::NoCalib,
        const std::string& path) {
        return true;
    }

    // **** Apply Calibration ****
    // Input[0]: Specifier for used method. (-> Struct for this in GrayCalibrationUtils.hpp)
    // Input[1]: const cv::Mat& -> image to apply calibration on.
    // Input[2]: const cv::Mat& -> Mask to show the valid pixels 
    // Output: cv::Mat -> the return value of the image distorted
    // -> Input[2]: This variable is mainly used for the LUT calibration since the LUT is build over the hole display
    // Modell/ ModellBias values are stored pixelwise cv::Mats - therefore a mask is not necessary but should still be provided
    // If Active Calibration is used -> The mask can be set to cv::Mat::ones(image.size(), CV_8U) or given as an empty cv::Mat.
    
    // Applies Active-Lut Calibration
    cv::Mat applyCalibration(
        const GrayCalibration_specifier::Active::LUT,
        const cv::Mat& image,
        cv::Mat& mask
    );

    // Applies Active-Model Calibration
    cv::Mat applyCalibration(
        const GrayCalibration_specifier::Active::Model,
        const cv::Mat& img,
        cv::Mat& mask
    );

    // Applies Active-Model-Bias Calibration
    cv::Mat applyCalibration(
        const GrayCalibration_specifier::Active::Model_Bias,
        const cv::Mat& img,
        cv::Mat& mask
    );

    // Applies Passive-Lut Calibratoin
    cv::Mat applyCalibration(
        const GrayCalibration_specifier::Passive::LUT,
        const cv::Mat& img,
        cv::Mat& mask
    );

    // Applies Passive-Model Calibration
    cv::Mat applyCalibration(
        const GrayCalibration_specifier::Passive::Model,
        const cv::Mat& img,
        cv::Mat& mask
    );

    // Applies Passive-Model-Bias
    cv::Mat applyCalibration(
        const GrayCalibration_specifier::Passive::Model_Bias,
        const cv::Mat& img,
        cv::Mat& mask
    );

    // Applies No Calibration -> directly return the input. 
    static cv::Mat applyCalibration(
        const GrayCalibration_specifier::NoCalib,
        const cv::Mat& img,
        const cv::Mat& mask)
    {
        return img;
    }


    // Input [1]: calib_specfier -> Must be struct that inherits from _Base::Specifier_Base (GrayCalibrationUtils.hpp)
    // Input [2]: const std::vector<cv::Mat>& images -> Vector of type cv::Mat of Type CV_64FC1
    // Input [3]: cv::Mat& mask -> Must be CV_8UC1
    // Output: bool value to show if Calibration was succesfull. 
    // The calibration Images or Lookup Table are stored in the ImageStoreClass -> (m_img_store)
    // The Calibration Frames are stored the following:
    // [0] -> Gamma
    // [1] -> I_max
    // [2] -> I_0 (Only Really used in fitGammaBias_LM, esle set to zero)
    // [3] -> Fitting Error R_2
    // [4] -> Used Iterations (Only Actively used in Gamma fitGammaBias_LM)
    // [5] -> Sample Imax val
    // [6] -> Sample Imin val
    // Destinction Between active and passive:
    // To simplify the procedures, essentially the same algorithm is used. 
    // - Passive (afterwards):
    // A mask and 256 gray images must be provided. Since the Mask already shows the allowed pixels no further action is needed
    // - Active (before Image Capturing):
    // Since here the Correspondense between Camera and Dispaly Pixel is needed, the 256 images must be evaluate beforehand.
    // For active Calibration therefore the images must have size of the display (in Pixels) and the mask is expected to be empty or cv::Mat::ones()
    // Since the Lut calibration uses the median over a the hole array the corropndence is not necessary.
	template <typename T> 
    bool doCalibration(
        T calib_specifier,
        const std::vector<cv::Mat>& images,
        cv::Mat& mask,
        const std::string& path)
    {
        // Produces hard error if wrong type is used for the type deduction. 
        static_assert(std::is_base_of<_Base::specifier_Base, T>::value);

        // If No calibration is selcted we directly return 
        if constexpr (std::is_same<T, GrayCalibration_specifier::NoCalib>::value) return true;

        CV_Assert(!path.empty());
        CV_Assert(!images.empty());
        CV_Assert(std::all_of(images.begin(), images.end(),
            [&](const cv::Mat& img) {
                return (images[0].size() == img.size()) &&
                    (img.type() == CV_64F) && (img.channels() == 1);
            }));
        // If NOT empty check if img & maks same size
        if (!mask.empty()) {
            CV_Assert(std::all_of(images.begin(), images.end(),
                [&](const cv::Mat& img) -> bool {
                    return img.size() == mask.size();
                }));
        }
        // If Mask is empty we create a new o
        else {
            mask = cv::Mat::ones(images[0].size(), CV_8U);
        }
        CV_Assert(images.size() == 256);
        CV_Assert(mask.type() == CV_8U);
        CV_Assert(m_impl != nullptr);

        // Case for Lut both active and passive
        if constexpr (std::is_same<T, GrayCalibration_specifier::Active::LUT>::value ||
           std::is_same<T, GrayCalibration_specifier::Passive::LUT>::value)
        {
           std::array<std::pair<double, double>, 256> lut =
                createLut(images, mask);

           m_img_store.add(FrameRole::GrayLUT, lut);

           m_img_store.saveLut(path);

           return true;
        }

        // All model Based Calibration Methods are created here.
        cv::Mat gamma(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
        cv::Mat Imax(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
        cv::Mat I_0(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
        cv::Mat errorR2(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
        cv::Mat iter(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));

        cv::Mat sample_imax(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
        cv::Mat sample_imin(mask.size(), CV_64FC1, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));

        // Parallel iterating over the rows 
        //cv::setNumThreads(0);
        cv::parallel_for_(cv::Range(0, mask.rows),
            [&](const cv::Range& range) {
                for (int r = range.start; r < range.end; ++r) {
                    // Create Pointers to the currently evaluated pixels 
                    const uchar* maskPtr = mask.ptr<uchar>(r);
                    double* gammaPtr = gamma.ptr<double>(r);
                    double* ImaxPtr = Imax.ptr<double>(r);
                    double* I_0Ptr = I_0.ptr<double>(r);
                    double* r2Ptr = errorR2.ptr<double>(r);
                    double* iterPtr = iter.ptr<double>(r);

                    double* imax_sample_ptr = sample_imax.ptr<double>(r);
                    double* imin_sample_ptr = sample_imin.ptr<double>(r);

                    for (int c = 0; c < mask.cols; ++c) {
                        if (maskPtr[c] == 0)continue;

                        // Extract an std::array<double, 256> array for each pixel from all the images. 
                        // The calibration is done for each pixel on these values.
                        std::array<double, 256> y{};
                        for (int i = 0; i < 256; ++i) {
                            y[i] = images[i].ptr<double>(r)[c];
                        }

                        auto minmaxit = std::minmax_element(y.begin(), y.end());

                        if (*minmaxit.first == *minmaxit.second) continue;

                        imax_sample_ptr[c] = *minmaxit.second;
                        imin_sample_ptr[c] = *minmaxit.first;

                        // This is checked at compile time. Therefore no expensive check is done in each iteration 
                        if constexpr (std::is_same<T, GrayCalibration_specifier::Active::Model>::value ||
                            std::is_same<T, GrayCalibration_specifier::Passive::Model>::value) 
                        {
                            auto res = fitGamma(y, 2.0, 1.0);
                            gammaPtr[c] = res.stats.gamma;
                            ImaxPtr[c] = res.stats.Imax;
                            I_0Ptr[c] = res.stats.I_0;
                            r2Ptr[c] = res.stats.r2;
                            iterPtr[c] = res.stats.iters;
                        }

                        else if constexpr (std::is_same<T, GrayCalibration_specifier::Active::Model_Bias>::value ||
                            std::is_same<T, GrayCalibration_specifier::Passive::Model_Bias>::value)
                        {
                            auto res = fitGammaBias_LM(y);
                            gammaPtr[c] = res.stats.gamma;
                            ImaxPtr[c] = res.stats.Imax;
                            I_0Ptr[c] = res.stats.I_0;
                            r2Ptr[c] = res.stats.r2;
                            iterPtr[c] = res.stats.iters;
                        }
                    }
                } 
            });

        // Save the images in Frame Role and directly save to .xml
        if constexpr (std::is_same<T, GrayCalibration_specifier::Active::Model>::value) {
            m_img_store.add(FrameRole::Modell_Active, gamma);
            m_img_store.add(FrameRole::Modell_Active, Imax);
            m_img_store.add(FrameRole::Modell_Active, I_0);
            m_img_store.add(FrameRole::Modell_Active, errorR2);
            m_img_store.add(FrameRole::Modell_Active, iter);
            m_img_store.add(FrameRole::Modell_Active, sample_imax);
            m_img_store.add(FrameRole::Modell_Active, sample_imin);
            // Save to path provided
            m_img_store.saveRoleXML(FrameRole::Modell_Active, path);
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Passive::Model>::value) {
            m_img_store.add(FrameRole::Modell_Passive, gamma);
            m_img_store.add(FrameRole::Modell_Passive, Imax);
            m_img_store.add(FrameRole::Modell_Passive, I_0);
            m_img_store.add(FrameRole::Modell_Passive, errorR2);
            m_img_store.add(FrameRole::Modell_Passive, iter);
            m_img_store.add(FrameRole::Modell_Passive, sample_imax);
            m_img_store.add(FrameRole::Modell_Passive, sample_imin);
            m_img_store.saveRoleXML(FrameRole::Modell_Passive, path);
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Active::Model_Bias>::value) {
            m_img_store.add(FrameRole::ModellBias_Active, gamma);
            m_img_store.add(FrameRole::ModellBias_Active, Imax);
            m_img_store.add(FrameRole::ModellBias_Active, I_0);
            m_img_store.add(FrameRole::ModellBias_Active, errorR2);
            m_img_store.add(FrameRole::ModellBias_Active, iter);
            m_img_store.add(FrameRole::ModellBias_Active, sample_imax);
            m_img_store.add(FrameRole::ModellBias_Active, sample_imin);
            m_img_store.saveRoleXML(FrameRole::ModellBias_Active, path);
        }
        else if constexpr (std::is_same<T, GrayCalibration_specifier::Passive::Model_Bias>::value) {
            m_img_store.add(FrameRole::ModellBias_Passive, gamma);
            m_img_store.add(FrameRole::ModellBias_Passive, Imax);
            m_img_store.add(FrameRole::ModellBias_Passive, I_0);
            m_img_store.add(FrameRole::ModellBias_Passive, errorR2);
            m_img_store.add(FrameRole::ModellBias_Passive, iter);
            m_img_store.add(FrameRole::ModellBias_Passive, sample_imax);
            m_img_store.add(FrameRole::ModellBias_Passive, sample_imin);
            m_img_store.saveRoleXML(FrameRole::ModellBias_Passive, path);
        }

        return true;
    }
};


#endif // !GRAYCALIB_HPP
