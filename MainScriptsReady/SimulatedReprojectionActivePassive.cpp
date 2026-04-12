#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>
#include "acquisitionworker.hpp"
#include <filesystem>
#include <regex>
#include "GrayCodeDecoder.hpp"
#include "screen.hpp"
#include "imgProcessing.hpp"
#include "CameraSimulation.hpp"

auto showNormalized2Channel = [](const cv::Mat& img, const std::string& winName = "Roflcopter")
    {
        CV_Assert(img.type() == CV_64FC2);

        std::vector<cv::Mat> channels(2);
        cv::split(img, channels);   // -> channels[0] = X, channels[1] = Y

        cv::Mat normX, normY;
        cv::normalize(channels[0], normX, 0, 255, cv::NORM_MINMAX);
        cv::normalize(channels[1], normY, 0, 255, cv::NORM_MINMAX);

        // Optional 3rd channel = zero for visualization
        cv::Mat zero = cv::Mat::zeros(img.rows, img.cols, CV_64F);

        cv::Mat merged;
        cv::merge(std::vector<cv::Mat>{normX, normY, zero}, merged); // 3-Channel Float

        cv::Mat display8u;
        merged.convertTo(display8u, CV_8UC3);

        cv::imshow(winName, display8u);
        cv::waitKey(0);
    };


int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-12_ReprojektionSimulated/QuantDispCamGamma2.2Lum_ActiveModelBiasNoAper" };

    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-11_Grauwertkalbirierung_Data_Simulated/2026-04-11_Sim_G2.2_nLumQuantDispApertureQantCam" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-09_StereoMAKO.xml" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();


    // Method used for calibration 
    _defl_::GrayCal::Method method = _defl_::GrayCal::Method::ActiveModel_Bias;

    meassure.setupCalibration(method, path_gray_calibration);

    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> distCoeffs = meassure.get(FrameRole::DistortionCoeff);

    GrayCodeConfig config{};
    config.creation.inverse = false;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 500;
    config.creation.resolution_y = 500;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode1 = pat.generateGrayCodeImg(config);

    std::size_t grayCodesz{ grayCode1.size() };

    std::vector<cv::Mat> sin_pattern = pat.generate_phaseShift<UniformRowsCols>(
        Shift_mode::four_phase_shift,
        10.0,
        127.5,
        127.5,
        1920,
        1080,
        {},
        true);

    // Active Calibration
    cv::Mat mask_c = cv::Mat::ones(sin_pattern[0].size(), CV_8U);

    int border = 50;

    // oben
    mask_c(cv::Range(0, border), cv::Range::all()) = 0;

    // unten
    mask_c(cv::Range(mask_c.rows - border, mask_c.rows), cv::Range::all()) = 0;

    // links
    mask_c(cv::Range::all(), cv::Range(0, border)) = 0;

    // rechts
    mask_c(cv::Range::all(), cv::Range(mask_c.cols - border, mask_c.cols)) = 0;

    for (auto& img : sin_pattern) {
        img = meassure.applyCalibration(img, mask_c, method);
    }

    std::size_t pattern_size{ sin_pattern.size() };

    std::vector<cv::Mat> pattern{ grayCode1 };

    for (const auto& img : sin_pattern) {
        pattern.push_back(img);
    }

    CameraSimulation simulation(processing);

    CameraSimulationConfig Sim_config;
    Sim_config.camera.camera_mat = camMatrix[0];
    Sim_config.camera.dist_coeffs = distCoeffs[0];
    Sim_config.scene.disp_shift_z = 4000.0;
    Sim_config.scene.disp_shift_x = -1920 / 2.0 * 0.2745;
    Sim_config.scene.disp_shift_y = -1080 / 2.0 * 0.2745;
    Sim_config.disp.scaling = 0.87;
    Sim_config.disp.bias = 0;

    Sim_config.scene.disp_tilt_x = 0;
    Sim_config.scene.disp_tilt_y = 0;
    Sim_config.scene.luminance = true;
    Sim_config.disp.gamma = 2.2;
    Sim_config.data.begin = pattern.begin()._Ptr;
    Sim_config.data.end = pattern.end()._Ptr;
    Sim_config.camera.apertureSmoothing = false;
    Sim_config.camera.f_number = 16.0;  // The phase has a wavelength of 108pix/2pi -> goal circl of confusion ~ 50pix ->  50mm/4 / 0.2745(pixepitch) = 45pix
    Sim_config.camera.circle_of_confusion_n_disp = 0;
    Sim_config.camera.quantization = true;
    Sim_config.disp.quantization = true;

    std::vector<cv::Mat> simulate =
        simulation.simulate(Sim_config);

    auto& eval = config.getEvalParameter();
    eval.start = simulate.begin();
    eval.end = std::next(simulate.begin(), grayCodesz);

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    cv::Mat mask = config.results.mask;

    // Just make the mask a bit smaller 
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);

    cv::erode(mask, mask, kernel, { -1,-1 }, 50);


    std::vector<cv::Mat> sin_pattern_sim(std::next(simulate.begin(), grayCodesz), simulate.end());

    // Passive Calib
    /*for (auto& img : sin_pattern_sim) {
        img = meassure.applyCalibration(img, mask, method);
    }*/

    std::vector<cv::Mat> wrapped_ref{ config };


    std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(sin_pattern_sim, 1, 4, true, path);

    std::vector<cv::Mat> unwrapVector;
    unwrapVector.push_back(wrapped[0]);
    unwrapVector.push_back(wrapped_ref[0]);
    unwrapVector.push_back(wrapped[1]);
    unwrapVector.push_back(wrapped_ref[1]);

    std::vector<cv::Mat> unwrap = meassure.do_unwrapped_phase(unwrapVector, mask, UnwrapMode::reference_Graycode, true, path, 108.0);

    std::vector<cv::Mat> reprojection = meassure.do_reprojection(
        unwrap,
        mask,
        camMatrix[0],
        distCoeffs[0],
        108.0,
        1,
        1,
        0.2745,
        true,
        path
    );


}

