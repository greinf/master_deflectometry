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

void saveRotationMatrix(const cv::Mat& rotMat, const std::string& path) {
    // Use .xml, .yml, or .json extension in the path
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "rotation_matrix" << rotMat;
    fs.release();
}

int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-05-07_GeometricCalibrationStereo_50" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
        "/data/2026-05-04_CameraStereoCalibration_k2_fix/Stereo_Calib.xml" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> distCoeffs = meassure.get(FrameRole::DistortionCoeff);

    meassure.load(FrameRole::CalibCamToCam, camMatrix_path);

    cv::Mat cam2camRot = meassure.get(FrameRole::CalibCamToCam)[0];
    cv::Mat cam2cam_tvec = meassure.get(FrameRole::CalibCamToCam)[1];

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
    config.creation.inverse = false;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 500;
    config.creation.resolution_y = 500;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode = pat.generateGrayCodeImg(config);

    std::size_t grayCodesz{ grayCode.size() };

    std::size_t sinPatternsz{ sin_pattern.size() };

    std::vector<cv::Mat> primary_Cam{ grayCode };

    for (const auto& img : sin_pattern) {
        primary_Cam.push_back(img);
    }

    std::size_t n_pics_prim{ primary_Cam.size() };

    int n_pics_per_val = 3;

    std::vector<cv::Mat> images = meassure.acquire_img(primary_Cam, FrameRole::Debug, n_pics_per_val);

    auto it = images.begin();

    std::vector<cv::Mat> meanimg;

    for (std::size_t i = 0; i < n_pics_prim; ++i) {
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

    // Mask defining the ROI
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

    // Secondary Camera part
    // Append the Full white and Full Black image at the end. 

    std::vector<cv::Mat> images_secondary = meassure.acquire_img(primary_Cam, FrameRole::Debug, n_pics_per_val);

    std::vector<cv::Mat> meanimg;

    for (std::size_t i = 0; i < n_pics_prim; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        meanimg.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    images_secondary = meanimg;


    auto& eval_sec = config.getEvalParameter();
    eval_sec.start = images.begin();
    eval_sec.end = std::next(images_secondary.begin(), grayCodesz);

    GrayCodeDecoder dec_sec(processing);

    dec_sec.decoding(config);

    std::vector<cv::Mat> pattern_sec(std::next(images_secondary.begin(), grayCodesz), images_secondary.end());

    // Mask defining the ROI
    cv::Mat mask_secon = config.results.mask;
    cv::Mat nanMask_sec = ~(mask_secon == mask_secon);
    mask_secon.setTo(0.0, nanMask_sec);

    std::vector<cv::Mat> wrapped_ref_sec{ config };

    std::vector<cv::Mat> wrapped_sec = meassure.do_wrapped_phase(pattern_sec, 1, 4, true, path + "/secondary");
    std::vector<cv::Mat> contrast_sec = meassure.get(FrameRole::Contrast);
    std::vector<cv::Mat> baseIntensity_sec = meassure.get(FrameRole::BaseIntensity);

    std::vector<cv::Mat> unwrapVector_sec;
    unwrapVector_sec.push_back(wrapped_sec[0]);
    unwrapVector_sec.push_back(wrapped_ref_sec[0]);
    unwrapVector_sec.push_back(wrapped_sec[1]);
    unwrapVector_sec.push_back(wrapped_ref_sec[1]);

    std::vector<cv::Mat> unwrap_sec = meassure.do_unwrapped_phase(unwrapVector_sec, mask_secon, UnwrapMode::reference_Graycode, true, path + "/secondary", 108.0);

    GeometricCalibrationConfig geoConfig{};
    geoConfig.displayPixelPitch = 0.2745;
    geoConfig.point_dist = 4;
    geoConfig.wavelength_phase = 108.0;
    geoConfig.pattern_size = cv::Size{ 20,20 };

    GeometricCalibration geoCalib(geoConfig, processing);

    GeometricCalibrationData_Stereo geodata;
    geodata.camMat = camMatrix[0];
    geodata.distCoeffs = distCoeffs[0];
    geodata.unwrap = &unwrap;
    geodata.contrast = &contrast;
    geodata.biasIntensity = &baseIntensity;
    geodata.mask_ROI = mask;
    geodata.path = path;

    /*cv::Mat cam2cam_RotErr = cam2camRot.t();
    cv::Mat cam2cam_tvecErr = -cam2cam_RotErr * cam2cam_tvec;*/
    geodata.mask_ROI_secon = mask_secon;
    geodata.camMat_secundaryCam = camMatrix[1];
    geodata.distCoeffs_secondaryCam = distCoeffs[1];
    geodata.cam2cam_rotMat = cam2camRot;
    geodata.cam2cam_tvec = cam2cam_tvec;
    geodata.biasIntensity_sec_cam = &baseIntensity_sec;
    geodata.contrast_sec_cam = &contrast_sec;

    //auto result = geoCalib.calibrateMono(geodata);

    auto result = geoCalib.calibrateStereo(geodata);

    auto& store = meassure.image_store();

    std::vector<cv::Mat> calib{ result.disp2cam_rvec.clone(), result.disp2cam_tvec.clone() };

    store.add(FrameRole::CalibDispToCam, calib);

    store.saveRoleXML(FrameRole::CalibDispToCam, path);

    cv::Mat Rotcam2mir;
    cv::Rodrigues(result.cam2mir_rvec, Rotcam2mir);

    saveRotationMatrix(Rotcam2mir, path + "/cam2mirRot.xml");

    cv::Mat mir2cam_tvec = -Rotcam2mir.t() * result.cam2mir_tvec;
    cv::Mat mir2cam_rvec;
    cv::Rodrigues(Rotcam2mir.t(), mir2cam_rvec);

    GeometricCalibrationTestData geotestData{ camMatrix[0], distCoeffs[0] };
    geotestData.unwrap = &unwrap;
    geotestData.contrast = &contrast;
    geotestData.biasIntensity = &baseIntensity;
    geotestData.disp2cam_rvec = result.disp2cam_rvec;
    geotestData.disp2cam_tvec = result.disp2cam_tvec;
    geotestData.mir2cam_rvec = mir2cam_tvec;
    geotestData.mir2cam_tvec = mir2cam_rvec;

    auto surfaces = geoCalib.test_calibration(geotestData);

    store.add(FrameRole::SurfaceNormals, surfaces);

    store.saveRoleXML(FrameRole::SurfaceNormals, path);

    std::cout << "Happy ? \n";

}
