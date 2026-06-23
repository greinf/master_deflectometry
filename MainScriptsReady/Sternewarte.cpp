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

void saveRotationMatrix(const cv::Mat& rotMat, const cv::Mat& tvec, const std::string& path) {
    // Use .xml, .yml, or .json extension in the path
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "rotation_matrix" << rotMat;
    fs << "tvec" << tvec;
    fs.release();
}

int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-05-16_SterenwarteTest" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
        "/data/2026-05-08_StereoCameraCalibration/Stereo_Calib.xml" };

    // 
    std::string cross_path{ "C:/Users/grein/Desktop/Fadenkreuz.png" };

    cv::Mat cross = cv::imread(cross_path, cv::IMREAD_GRAYSCALE);
    cv::Mat cross_resize;
    cv::resize(cross, cross_resize, { 1920, 1080 });

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();
    ImageStore& img_store = meassure.image_store();

    //meassure.getFrames(FrameRole::Debug, 1, cross_resize);

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

    /*Triangulate triangulate{
        camMatPrim,
        distCoeffsPrim,
        camMatSecon,
        distCoeffsSecon,
        rot,
        trans };

    cv::Point2f prim(1055, 943);
    cv::Point2f secon(948, 1091);

    cv::Point3f trinagulated_pt = triangulate.calculate(prim, secon);

    triangulate.savePoint(path + "/triangulated.csv", prim, secon, trinagulated_pt);*/

    GrayCodeConfig config{};
    config.creation.inverse = false;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 20;
    config.creation.resolution_y = 20;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode1 = pat.generateGrayCodeImg(config);

    std::size_t grayCodesz{ grayCode1.size() };

    std::vector<cv::Mat> sin_pattern = pat.generate_phaseShift<UniformRowsCols>(
        Shift_mode::four_phase_shift,
        10.0 * 5,
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

    int n_pics_per_val = 3;

    std::vector<cv::Mat> images_prim = meassure.acquire_img(pattern, FrameRole::RawInput, n_pics_per_val);

    // save Raw Primary Input
    img_store.saveRole(FrameRole::RawInput, path + "/primary");

    std::vector<cv::Mat> mean_img_prim;

    std::vector<cv::Mat>::iterator it = images_prim.begin();

    for (std::size_t i = 0; i < n_pics; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        mean_img_prim.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    auto& eval = config.getEvalParameter();
    eval.start = mean_img_prim.begin();
    eval.end = std::next(mean_img_prim.begin(), grayCodesz);

    std::vector<cv::Mat> patternd(std::next(mean_img_prim.begin(), grayCodesz), mean_img_prim.end());

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    std::vector<cv::Mat> wrapped_ref{ config };

    std::vector<cv::Mat> wrapped = meassure.do_wrapped_phase(patternd, 1, 4, true, path + "/primary");

    std::vector<cv::Mat> contrast = meassure.get(FrameRole::Contrast);
    std::vector<cv::Mat> biasIntensity = meassure.get(FrameRole::BaseIntensity);

    std::vector<cv::Mat> amplitude(2);

    cv::multiply(contrast[0], biasIntensity[0], amplitude[0]);
    cv::multiply(contrast[1], biasIntensity[1], amplitude[1]);

    cv::Mat mask_prim = processing.createMask(amplitude, 0.5, true);

    img_store.add(FrameRole::mask, mask_prim);
    img_store.saveRole(FrameRole::mask, path + "/primary");

    img_store.clear(FrameRole::mask);

    std::vector<cv::Mat> unwrapVector;
    unwrapVector.push_back(wrapped[0]);
    unwrapVector.push_back(wrapped_ref[0]);
    unwrapVector.push_back(wrapped[1]);
    unwrapVector.push_back(wrapped_ref[1]);

    std::vector<cv::Mat> unwrap =
        meassure.do_unwrapped_phase(unwrapVector, mask_prim, UnwrapMode::reference_Graycode, true, path + "/primary", 108.0 / 5.0);

    // Secondary Camera part. 
    std::vector<cv::Mat> images_secondary = meassure.acquire_img(pattern, FrameRole::RawInput, n_pics_per_val);

    img_store.saveRole(FrameRole::RawInput, path + "/secondary");

    it = images_secondary.begin();

    std::vector<cv::Mat> mean_secondary;

    for (std::size_t i = 0; i < n_pics; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        mean_secondary.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    eval = config.getEvalParameter();
    eval.start = mean_secondary.begin();
    eval.end = std::next(mean_secondary.begin(), grayCodesz);

    // Sinus Pattern
    std::vector<cv::Mat> patternd_secon(std::next(mean_secondary.begin(), grayCodesz), mean_secondary.end());

    dec.decoding(config);

    std::vector<cv::Mat> wrapped_ref_secon{ config };

    images_secondary = mean_secondary;

    std::vector<cv::Mat> images_secondary_wrapped = meassure.do_wrapped_phase(patternd_secon, 1, 4, true, path + "/secondary");
    std::vector<cv::Mat> contrast_secondary = meassure.get(FrameRole::Contrast);
    std::vector<cv::Mat> baseIntensity_secondary = meassure.get(FrameRole::BaseIntensity);

    std::vector<cv::Mat> amplitude_secon(2);

    cv::multiply(contrast_secondary[0], baseIntensity_secondary[0], amplitude_secon[0]);
    cv::multiply(contrast_secondary[1], baseIntensity_secondary[1], amplitude_secon[1]);


    cv::Mat mask_secon = processing.createMask(amplitude_secon, 0.5, true);

    std::vector<cv::Mat> unwrap_vector_secon(4);
    unwrap_vector_secon[0] = images_secondary_wrapped[0];
    unwrap_vector_secon[1] = wrapped_ref_secon[0];
    unwrap_vector_secon[2] = images_secondary_wrapped[1];
    unwrap_vector_secon[3] = wrapped_ref_secon[1];

    std::vector<cv::Mat> unwrap_secondary =
        meassure.do_unwrapped_phase(unwrap_vector_secon, mask_secon, UnwrapMode::reference_Graycode, true, path + "/secondary", 108.0 / 5.0);

    // PSF part
    int bins = 10;

    std::vector<cv::Mat> psf_prim, psf_secon;

    do {
        // PSF berechnen
        PSF_Config psf_config{
            108.0 / 5.0,
            bins,
            {1920, 1080}
        };

        PSF_Data psf_data{};
        psf_data.mask = &mask_prim;
        psf_data.unwrap = &unwrap;

        PSF psf_eval(psf_config);

        auto result = psf_eval.computePSF(psf_data);

        // anzeigen
        shownormalized(result.histo);

        std::cout << "\nCurrent bin size: " << psf_config.m_bins << std::endl;
        std::cout << "Enter new bin size (0 = keep current): ";

        std::cin >> bins;

        psf_prim.push_back(result.histo);

    } while (bins != 0);

    img_store.clear(FrameRole::PSFgt);
    //img_store.clear(FrameRole::PSF);
    img_store.add(FrameRole::PSFgt, psf_prim);
    //img_store.add(FrameRole::PSF, psf_images_prim);
    img_store.saveRole(FrameRole::PSFgt, path + "primary");
    //img_store.saveRole(FrameRole::PSF, path);

    bins = 10;
    do {
        // PSF berechnen
        PSF_Config psf_config{
            108.0 / 5.0,
            bins,
            {1920, 1080}
        };

        PSF_Data psf_data{};
        psf_data.mask = &mask_secon;
        psf_data.unwrap = &unwrap_secondary;

        PSF psf_eval(psf_config);

        auto result = psf_eval.computePSF(psf_data);

        // anzeigen
        shownormalized(result.histo);

        std::cout << "\nCurrent bin size: " << psf_config.m_bins << std::endl;
        std::cout << "Enter new bin size (0 = keep current): ";

        std::cin >> bins;

        psf_secon.push_back(result.histo);

    } while (bins != 0);

    /* std::cout << "Primary Camera \n";
     std::vector<cv::Mat> psf_images_prim;
     for (auto& img : psf_prim) {
         cv::Mat norm;
         cv::normalize(img, norm, 0, 255.0, cv::NORM_MINMAX, CV_8U);
         std::vector<cv::Mat> img = meassure.getFrames(FrameRole::PSF, 1, norm);
         psf_images_prim.push_back(img[0]);
     }*/


     /*std::cout << "Secondary Camera \n";
     std::vector<cv::Mat> psf_images_secon;
     for (auto& img : psf_secon) {
         cv::Mat norm;
         cv::normalize(img, norm, 0, 255.0, cv::NORM_MINMAX, CV_8U);
         std::vector<cv::Mat> img = meassure.getFrames(FrameRole::PSF, 1, norm);
         psf_images_secon.push_back(img[0]);
     }*/

    img_store.clear(FrameRole::PSFgt);
    //img_store.clear(FrameRole::PSF);
    img_store.add(FrameRole::PSFgt, psf_secon);
    //img_store.add(FrameRole::PSF, psf_images_secon);
    img_store.saveRole(FrameRole::PSFgt, path + "/secondary");
    //img_store.saveRole(FrameRole::PSF, path + "/secondary");

}
