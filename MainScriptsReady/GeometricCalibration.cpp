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
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-21_GeometricCalibration_SurfaceNormals" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
        "/data/2026-04-13_CameraCalibration/Mono_Calib.xml" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> distCoeffs = meassure.get(FrameRole::DistortionCoeff);

    //std::vector<cv::Mat> img = meassure.getFrames(FrameRole::Debug, 1);

    std::vector<cv::Mat> sin_pattern = pat.generate_phaseShift<UniformRowsCols>(
        Shift_mode::user_defined,
        10.0,
        127.5,
        127.5,
        1920,
        1080,
        {}
    );

    GrayCodeConfig config{};
    config.creation.inverse = true;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 1000;
    config.creation.resolution_y = 1000;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode = pat.generateGrayCodeImg(config);

    std::size_t grayCodesz{ grayCode.size() };

    std::size_t sinPatternsz{ sin_pattern.size() };

    std::vector<cv::Mat> sim_inputFrames{ grayCode };

    for (const auto& img : sin_pattern) {
        sim_inputFrames.push_back(img);
    }

    std::size_t n_pics = grayCodesz + sinPatternsz;

    int n_pics_per_val = 4;

    std::vector<cv::Mat> images = meassure.acquire_img(sim_inputFrames, FrameRole::Debug, n_pics_per_val);

    auto it = images.begin();

    std::vector<cv::Mat> meanimg;

    for (std::size_t i = 0; i < n_pics; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        meanimg.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    images = meanimg;

    auto& eval = config.getEvalParameter();
    eval.start = images.begin();
    eval.end = std::next(images.begin(), grayCodesz);

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    std::vector<cv::Mat> patternd(std::next(images.begin(), grayCodesz), images.end());

    cv::Mat mask = config.results.mask;

    cv::Mat nanMask = ~(mask == mask);
    mask.setTo(0.0, nanMask);

    std::vector<cv::Mat> wrapped_ref{ config };

    std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(patternd, 1, 50, true, path);
    std::vector<cv::Mat> contrast = meassure.get(FrameRole::Contrast);
    std::vector<cv::Mat> baseIntensity = meassure.get(FrameRole::BaseIntensity);

    std::vector<cv::Mat> unwrapVector;
    unwrapVector.push_back(wrapped[0]);
    unwrapVector.push_back(wrapped_ref[0]);
    unwrapVector.push_back(wrapped[1]);
    unwrapVector.push_back(wrapped_ref[1]);

    std::vector<cv::Mat> unwrap = meassure.do_unwrapped_phase(unwrapVector, mask, UnwrapMode::reference_Graycode, true, path, 108.0);

    GeometricCalibrationConfig geoConfig{};
    geoConfig.displayPixelPitch = 0.2745;
    geoConfig.point_dist = 4;
    geoConfig.wavelength_phase = 108.0;
    geoConfig.pattern_size = cv::Size{ 20,20 };

    GeometricCalibration geoCalib(geoConfig, processing);
    GeometricCalibrationData geodata{ camMatrix[0], distCoeffs[0] };
    geodata.unwrap = &unwrap;
    geodata.contrast = &contrast;
    geodata.biasIntensity = &baseIntensity;
    geodata.mask = mask;

    auto result = geoCalib.calibrate(geodata);

    auto& store = meassure.image_store();

    std::vector<cv::Mat> calib{ result.disp2cam_rvec, result.disp2cam_tvec };

    store.add(FrameRole::CalibDispToCam, calib);

    store.saveRoleXML(FrameRole::CalibDispToCam, path);

    GeometricCalibrationTestData geotestData{ camMatrix[0], distCoeffs[0] };
    geotestData.unwrap = &unwrap;
    geotestData.contrast = &contrast;
    geotestData.biasIntensity = &baseIntensity;
    geotestData.disp2cam_rvec = result.disp2cam_rvec;
    geotestData.disp2cam_tvec = result.disp2cam_tvec;
    geotestData.cam2mir_rvec = result.cam2mir_rvec;
    geotestData.cam2mir_tvec = result.cam2mir_tvec;

    auto surfaces = geoCalib.test_calibration(geotestData);

    store.add(FrameRole::SurfaceNormals, surfaces);

    store.saveRoleXML(FrameRole::SurfaceNormals, path);

    std::cout << "Happy ? \n";

}
    //////// ************ Triangulation *******************
////
////    //auto result = meassure.computeLaserDistanceFrom4Images(
////    //    LaserFr[0], LaserFr[1], LaserFr[2], LaserFr[3],
////    //    camMatrix[0], dist_Coeffs[0], camMatrix[1], dist_Coeffs[1], camToCam[0], camToCam[1]
////    //);
////
////    //std::cout << "p1 (px): " << result.p1_px << "\n";
////    //std::cout << "p2 (px): " << result.p2_px << "\n";
////    //std::cout << "3D (cam1): [" << result.X_cam1.x << ", " << result.X_cam1.y << ", " << result.X_cam1.z << "]\n";
////    //std::cout << "distance to cam1: " << result.distance_cam1 << " (same units as T)\n";


    // AbstandsMessung 
    // meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    // std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);

    // meassure.load(FrameRole::CalibCamToCam, camMatrix_path);
    // std::vector<cv::Mat> camToCam = meassure.get(FrameRole::CalibCamToCam);

    // std::vector<cv::Mat> dist_Coeffs = meassure.get(FrameRole::DistortionCoeff);


    // std::vector<cv::Mat> pattern_distorted;
    // for (auto& img : unwrappedPhase) {
    //     pattern_distorted.emplace_back(meassure.distortImage_manual(img, camMatrix[0], dist_Coeffs[0]));
    // }



