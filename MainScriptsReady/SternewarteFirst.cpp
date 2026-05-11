#define VERSION "1"

// 05.05.2026 SternenwarteSkript 
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
#include "PSF.hpp"
#include "Triangulate.hpp"


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
    cv::destroyWindow("norm");
    };


int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    //!!!!!!!!!!!!!!! SavePath !!!!!!!!!!!!!
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-05-05_50PhaseShiftRun_8*Pattern" };

    // Triangualte Point
    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
         "/data/2026-05-04_CameraStereoCalibration_k2_fix/Stereo_Calib.xml" };

    Deflectometry meassure{};

    //meassure.getFrames(FrameRole::Debug, 1);
    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();
    ImageStore& img_store = meassure.image_store();

    // !!!!!!!!!! Triangulation Part !!!!!! 

    img_store.loadCalibrationMatrix(camMatrix_path);
    img_store.loadCalibCamToCam(camMatrix_path);

    std::vector<cv::Mat> intrinsic_calib = img_store.get(FrameRole::CalibrationMatrix);

    std::vector<cv::Mat> extrinsic_Calib = img_store.get(FrameRole::CalibCamToCam);

    std::vector<cv::Mat> distCoeffs = img_store.get(FrameRole::DistortionCoeff);
    cv::Mat rot = extrinsic_Calib[0];
    cv::Mat trans = extrinsic_Calib[1];

    cv::Mat camMatPrim = intrinsic_calib[0];
    cv::Mat camMatSecon = intrinsic_calib[1];
    cv::Mat distCoeffsPrim = distCoeffs[0];
    cv::Mat distCoeffsSecon = distCoeffs[1];

    Triangulate triangulate{
        camMatPrim,
        distCoeffsPrim,
        camMatSecon,
        distCoeffsSecon,
        rot,
        trans};

    cv::Point2f prim(1351, 877);
    cv::Point2f secon(1080, 1136);

    cv::Point3f trinagulated_pt = triangulate.calculate(prim, secon);

    triangulate.savePoint(path + "/triangulated.csv", prim, secon, trinagulated_pt);

    GrayCodeConfig config{};
    config.creation.inverse = true;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 500;
    config.creation.resolution_y = 500;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode1 = pat.generateGrayCodeImg(config);

    // meassure.getFrames(FrameRole::Debug, 1);

    std::size_t grayCodesz{ grayCode1.size() };

    std::vector<cv::Mat> sin_pattern = pat.generate_phaseShift<UniformRowsCols>(
        Shift_mode::user_defined,
        10.0 * 8,
        127.5,
        127.5,
        1920,
        1080,
        {});

    std::size_t pattern_size{ sin_pattern.size() };

    std::vector<cv::Mat> pattern{ grayCode1 };

    for (const auto& img : sin_pattern) {
        pattern.push_back(img);
    }

    std::size_t n_pics = grayCodesz + pattern_size;

    int n_pics_per_val = 1;

    std::vector<cv::Mat> images = meassure.acquire_img(pattern, FrameRole::Debug, n_pics_per_val);

    std::vector<cv::Mat> mean_img;

    std::vector<cv::Mat>::iterator it = images.begin();

    for (std::size_t i = 0; i < n_pics; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        mean_img.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    auto& eval = config.getEvalParameter();
    eval.start = mean_img.begin();
    eval.end = std::next(mean_img.begin(), grayCodesz);

    std::vector<cv::Mat> patternd(std::next(mean_img.begin(), grayCodesz), mean_img.end());

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    std::vector<cv::Mat> wrapped_ref{ config };

    std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(patternd, 1, 50, true, path);

    std::vector<cv::Mat> contrast = meassure.get(FrameRole::Contrast);

    cv::Mat mask = processing.createMask(contrast, 0.3, true);

    std::vector<cv::Mat> unwrapVector;
    unwrapVector.push_back(wrapped[0]);
    unwrapVector.push_back(wrapped_ref[0]);
    unwrapVector.push_back(wrapped[1]);
    unwrapVector.push_back(wrapped_ref[1]);

    std::vector<cv::Mat> unwrap = meassure.do_unwrapped_phase(unwrapVector, mask, UnwrapMode::reference_Graycode, true, path, 108.0 / 8.0);

    // Secondary Camera part. 
    std::vector<cv::Mat> images_secondary = meassure.acquire_img(sin_pattern, FrameRole::Debug, n_pics_per_val);

    it = images_secondary.begin();

    std::vector<cv::Mat> mean_secondary;

    for (std::size_t i = 0; i < pattern_size; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        mean_secondary.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    images_secondary = mean_secondary;

    std::vector<cv::Mat> images_secondary_warpped = meassure.do_wrapped_phase(images_secondary, 1, 50, true, path + "/secondary/");
    std::vector<cv::Mat> contrast_secondary = meassure.get(FrameRole::Contrast);
    std::vector<cv::Mat> baseIntensity_secondary = meassure.get(FrameRole::BaseIntensity);

    int bins = 10;

    std::vector<cv::Mat> psf;

    do {
        // PSF berechnen
        PSF_Config psf_config{
            108.0,
            bins,
            {1920, 1080}
        };

        PSF_Data psf_data{};
        psf_data.mask = &mask;
        psf_data.unwrap = &unwrap;

        PSF psf_eval(psf_config);

        auto result = psf_eval.computePSF(psf_data);

        // anzeigen
        shownormalized(result.histo);

        std::cout << "\nCurrent bin size: " << psf_config.m_bins << std::endl;
        std::cout << "Enter new bin size (0 = keep current): ";

        std::cin >> bins;

        psf.push_back(result.histo);

    } while (bins != 0);

    for (auto& img : psf) {
        cv::Mat norm;
        cv::normalize(img, norm, 0, 255.0, cv::NORM_MINMAX, CV_8U);
    }

    std::vector<cv::Mat> images_psf = meassure.acquire_img(psf, FrameRole::Debug, 1);

    img_store.add(FrameRole::PSFgt, psf);

    img_store.add(FrameRole::PSF, images_psf);

    img_store.saveRole(FrameRole::PSF, path);

    img_store.saveRole(FrameRole::PSFgt, path);

}
