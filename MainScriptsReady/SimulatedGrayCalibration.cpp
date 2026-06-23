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
    
    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-11_Sim_G2.2_nDispQuant" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-09_StereoMAKO.xml" };

    Deflectometry meassure{};

    //meassure.GrayCalibrationClassTest(_defl_::GrayCal::Method::ActiveModel_Bias, path_gray_calibration);

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();


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

    std::vector<cv::Mat> patterngray = pat.generateGrayCalibrationSequence(1);

    //std::vector<cv::Mat> pattern_gamma = processing.do_gamma_distortion(2.2, patterngray);

    std::size_t pattern_size{ patterngray.size() };

    std::vector<cv::Mat> pattern{ grayCode1 };

    for (const auto& img : patterngray) {
        pattern.push_back(img);
    }

    CameraSimulation simulation(processing);

    CameraSimulationConfig Sim_config;
    Sim_config.camera.camera_mat = camMatrix[0];
    Sim_config.camera.dist_coeffs = distCoeffs[0];
    Sim_config.scene.disp_shift_z = 4000.0;
    Sim_config.scene.disp_shift_x = -1920 / 2.0 * 0.2745;
    Sim_config.scene.disp_shift_y = -1080 / 2.0 * 0.2745;

    Sim_config.scene.disp_tilt_x = 0;
    Sim_config.scene.disp_tilt_y = 0;
    Sim_config.scene.luminance = false;
    Sim_config.disp.gamma = 2.2;
    Sim_config.data.begin = pattern.begin()._Ptr;
    Sim_config.data.end = pattern.end()._Ptr;
    Sim_config.camera.apertureSmoothing = false;
    Sim_config.camera.circle_of_confusion_n_disp = 0;
    Sim_config.camera.quantization = false;
    Sim_config.disp.quantization = true;

    std::vector<cv::Mat> simulate =
        simulation.simulate(Sim_config);

    auto& eval = config.getEvalParameter();
    eval.start = simulate.begin();
    eval.end = std::next(simulate.begin(), grayCodesz);

    cv::Mat mask = config.results.mask;

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    cv::Mat homography =
        processing.createHomographyFromGrayCode(config, cv::Size(1920, 1080));

    std::vector<cv::Mat> maps = processing.createMappingfromHomography(homography, cv::Size(1920, 1080));

    std::vector<cv::Mat> grayValues_simulated(std::next(simulate.begin(), grayCodesz), simulate.end());


    std::vector<cv::Mat> grayValues_mapped = processing.remapCameraToScreen(grayValues_simulated, maps);

    //meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveModel);

    // meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveLut);

    meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::LocLutPassive);

    meassure.do_grayvalue_calibration(grayValues_simulated, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveModel);

    meassure.do_grayvalue_calibration(grayValues_simulated, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveLut);

    meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveModel_Bias);

    meassure.do_grayvalue_calibration(grayValues_simulated, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveModel_Bias);

}