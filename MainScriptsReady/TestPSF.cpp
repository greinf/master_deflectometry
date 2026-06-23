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
#include "CameraCalibration.hpp"
#include "GeometricCalibration.hpp"
#include "PSF.hpp"

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

auto shownormalized = [](const cv::Mat& d) {
    cv::Mat img;
    cv::normalize(d, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("norm", img);
    cv::waitKey(0);
    };

int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/TESTPSF" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
        "/data/2026-04-25_CameraStereoCalibration/Stereo_Calib.xml" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> distCoeffs = meassure.get(FrameRole::DistortionCoeff);

    std::vector<cv::Mat> sin_pattern = pat.generate_phaseShift<UniformRowsCols>(
        Shift_mode::four_phase_shift,
        10.0,
        127.5,
        127.5,
        1920,
        1080,
        {}
    );

    GrayCodeConfig config{};
    config.creation.inverse = false;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 500;
    config.creation.resolution_y = 500;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode = pat.generateGrayCodeImg(config);

    std::size_t grayCodesz{ grayCode.size() };

    std::size_t sinPatternsz{ sin_pattern.size() };

    std::vector<cv::Mat> sim_inputFrames{ grayCode };

    for (const auto& img : sin_pattern) {
        sim_inputFrames.push_back(img);
    }

    CameraSimulation simulation(processing);

    CameraSimulationConfig Sim_config;
    Sim_config.camera.camera_mat = camMatrix[1];
    Sim_config.camera.dist_coeffs = distCoeffs[1];
    Sim_config.mirror.contains_mirror = true;
    Sim_config.mirror.mirror_sz = cv::Size(400, 400);
    Sim_config.scene.disp_shift_z = 0.0;
    Sim_config.scene.disp_shift_x = -1920 / 2.0 * 0.2745;
    Sim_config.scene.disp_shift_y = -1080 / 2.0 * 0.2745;
    Sim_config.disp.scaling = 1.0;
    Sim_config.disp.bias = 0;
    Sim_config.scene.disp_tilt_x = 0;
    Sim_config.scene.disp_tilt_y = 0;
    Sim_config.scene.luminance = false;
    Sim_config.disp.gamma = 1.0;
    Sim_config.scene.mirror_shift_x = -200.0;
    Sim_config.scene.mirror_shift_y = -200.0;
    Sim_config.scene.mirror_shift_z = 2500.0;
    Sim_config.scene.mirror_tilt_x = 0.00;
    Sim_config.scene.mirror_tilt_y = 0.00;
    Sim_config.data.begin = sim_inputFrames.begin()._Ptr;
    Sim_config.data.end = sim_inputFrames.end()._Ptr;
    Sim_config.camera.apertureSmoothing = false;
    Sim_config.camera.f_number = 16.0;  // The phase has a wavelength of 108pix/2pi -> goal circl of confusion ~ 50pix ->  50mm/4 / 0.2745(pixepitch) = 45pix
    Sim_config.camera.circle_of_confusion_n_disp = 0;
    Sim_config.camera.quantization = false;
    Sim_config.disp.quantization = false;
    Sim_config.scene.disp_flip_vertical = true;

    std::vector<cv::Mat> simulate =
        simulation.simulate(Sim_config);

    auto& eval = config.getEvalParameter();
    eval.start = simulate.begin();
    eval.end = std::next(simulate.begin(), grayCodesz);

    std::vector<cv::Mat> patternd(std::next(simulate.begin(), grayCodesz), simulate.end());

    GrayCodeDecoder dec(processing);
    dec.decoding(config);

    cv::Mat mask = config.results.mask;

    std::vector<cv::Mat> wrapped_ref{ config };

    std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(patternd, 1, 4, false, path);
    std::vector<cv::Mat> contrast = meassure.get(FrameRole::Contrast);
    std::vector<cv::Mat> baseIntensity = meassure.get(FrameRole::BaseIntensity);

    std::vector<cv::Mat> unwrapVector;
    unwrapVector.push_back(wrapped[0]);
    unwrapVector.push_back(wrapped_ref[0]);
    unwrapVector.push_back(wrapped[1]);
    unwrapVector.push_back(wrapped_ref[1]);

    std::vector<cv::Mat> unwrap = meassure.do_unwrapped_phase(unwrapVector, mask, UnwrapMode::reference_Graycode, false, path, 108.0);

    PSF_Config psf_config{ 108.0, 10, {1920, 1080} };

    PSF_Data psf_data{};
    psf_data.mask = &mask;
    psf_data.unwrap = &unwrap;

    PSF psf{ psf_config };

    auto result = psf.computePSF(psf_data);

    std::cout << "Happy \n";
}