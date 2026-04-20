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
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-13_ReprojektionReal/ActiveLut_ScaledSig" };

    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-13_Grauwertkalbirierung_Data_Real/2026-04-13_dark_room" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
        "/data/2026-04-13_CameraCalibration/Mono_Calib.xml" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    // Method used for calibration 
    _defl_::GrayCal::Method method = _defl_::GrayCal::Method::PassiveLut;

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

    //// Active Calibration
    cv::Mat mask_c = cv::Mat::ones(sin_pattern[0].size(), CV_8U);

    //int border = 10;

    //// oben
    //mask_c(cv::Range(0, border), cv::Range::all()) = 0;

    //// unten
    //mask_c(cv::Range(mask_c.rows - border, mask_c.rows), cv::Range::all()) = 0;

    //// links
    //mask_c(cv::Range::all(), cv::Range(0, border)) = 0;

    //// rechts
    //mask_c(cv::Range::all(), cv::Range(mask_c.cols - border, mask_c.cols)) = 0;

    //for (auto& img : sin_pattern) {
    //    img = meassure.applyCalibration(img, mask_c, method);
    //}

    std::size_t pattern_size{ sin_pattern.size() };

    std::vector<cv::Mat> pattern{ grayCode1 };

    for (const auto& img : sin_pattern) {
        pattern.push_back(img);
    }

    /*std::size_t n_pics = grayCodesz + patternSize;*/

    int n_pics_per_val = 1;

    std::vector<cv::Mat> images = meassure.acquire_img(pattern, FrameRole::Debug, n_pics_per_val);

    // auto it = images.begin();

    auto& eval = config.getEvalParameter();
    eval.start = images.begin();
    eval.end = std::next(images.begin(), grayCodesz);

    std::vector<cv::Mat> patternd(std::next(images.begin(), grayCodesz), images.end());

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    cv::Mat mask = config.results.mask;
    // Just make the mask a bit smaller 
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);

    cv::erode(mask, mask, kernel, { -1,-1 }, 20);


    // Passive Calib
    for (auto& img : patternd) {
        img = meassure.applyCalibration(img, mask, method);
    }


    std::vector<cv::Mat> wrapped_ref{ config };

    std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(patternd, 1, 4, true, path);

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

    /*std::vector<cv::Mat> mean_img(n_pics);

    for (std::size_t i = 0; i < n_pics; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        mean_img[i] = processing.mean(std::vector<cv::Mat>(it, it_end));
        it = it_end;
    }

    images = mean_img;*/



    //cv::Mat homography =
    //    processing.createHomographyFromGrayCode(config, cv::Size(1920, 1080));

    //std::vector<cv::Mat> maps = processing.createMappingfromHomography(homography, cv::Size(1920, 1080));

    //std::vector<cv::Mat> grayValues(std::next(images.begin(), grayCodesz), images.end());

    //std::vector<cv::Mat> grayValues_mapped = processing.remapCameraToScreen(grayValues, maps);

    //meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveModel);

    ////meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveLut);

    //meassure.do_grayvalue_calibration(grayValues, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveModel);

    //meassure.do_grayvalue_calibration(grayValues, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveLut);

    //meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveModel_Bias);

    //meassure.do_grayvalue_calibration(grayValues, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveModel_Bias);



    //// Active Calibration
    //cv::Mat mask_c = cv::Mat::ones(grayValues[0].size(), CV_8U);

    //int border = 50;

    //// oben
    //mask_c(cv::Range(0, border), cv::Range::all()) = 0;

    //// unten
    //mask_c(cv::Range(mask_c.rows - border, mask_c.rows), cv::Range::all()) = 0;

    //// links
    //mask_c(cv::Range::all(), cv::Range(0, border)) = 0;

    //// rechts
    //mask_c(cv::Range::all(), cv::Range(mask_c.cols - border, mask_c.cols)) = 0;

    //for (auto& img : sin_pattern) {
    //    img = meassure.applyCalibration(img, mask_c, method);
    //}

    //cv::Mat mask = config.results.mask;

    //// Just make the mask a bit smaller 
    //cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);

    //cv::erode(mask, mask, kernel, { -1,-1 }, 50);


    //std::vector<cv::Mat> sin_pattern_sim(std::next(simulate.begin(), grayCodesz), simulate.end());

    //// Passive Calib
    ///*for (auto& img : sin_pattern_sim) {
    //    img = meassure.applyCalibration(img, mask, method);
    //}*/

    //std::vector<cv::Mat> wrapped_ref{ config };
    //

    //std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(sin_pattern_sim, 1, 4, true, path);
    //
    //std::vector<cv::Mat> unwrapVector;
    //unwrapVector.push_back(wrapped[0]);
    //unwrapVector.push_back(wrapped_ref[0]);
    //unwrapVector.push_back(wrapped[1]);
    //unwrapVector.push_back(wrapped_ref[1]);

    //std::vector<cv::Mat> unwrap = meassure.do_unwrapped_phase(unwrapVector, mask, UnwrapMode::reference_Graycode, true, path, 108.0);

    //std::vector<cv::Mat> reprojection = meassure.do_reprojection(
    //    unwrap,
    //    mask,
    //    camMatrix[0],
    //    distCoeffs[0],
    //    108.0,
    //    1,
    //    1,
    //    0.2745, 
    //    true,
    //    path
    //);

   // AbstandsMessung 
   // meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
   // std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);

   // meassure.load(FrameRole::CalibCamToCam, camMatrix_path);
   // std::vector<cv::Mat> camToCam = meassure.get(FrameRole::CalibCamToCam);

   // std::vector<cv::Mat> dist_Coeffs = meassure.get(FrameRole::DistortionCoeff);

   // std::vector<cv::Vec2d> ref_point =
   //     meassure.getReferencePoint(
   //         std::vector<cv::Mat>{reference_img},
   //         ReferenceMode::checkerboard, 
   //         mask);

   // std::vector<cv::Mat> pattern_distorted;
   // for (auto& img : unwrappedPhase) {
   //     pattern_distorted.emplace_back(meassure.distortImage_manual(img, camMatrix[0], dist_Coeffs[0]));
   // }

}

//// ************ Triangulation *******************

    //auto result = meassure.computeLaserDistanceFrom4Images(
    //    LaserFr[0], LaserFr[1], LaserFr[2], LaserFr[3],
    //    camMatrix[0], dist_Coeffs[0], camMatrix[1], dist_Coeffs[1], camToCam[0], camToCam[1]
    //);

    //std::cout << "p1 (px): " << result.p1_px << "\n";
    //std::cout << "p2 (px): " << result.p2_px << "\n";
    //std::cout << "3D (cam1): [" << result.X_cam1.x << ", " << result.X_cam1.y << ", " << result.X_cam1.z << "]\n";
    //std::cout << "distance to cam1: " << result.distance_cam1 << " (same units as T)\n";
