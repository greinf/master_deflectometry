#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>
#include "acquisitionworker.hpp"
#include <filesystem>





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

auto showNormalized = [](const cv::Mat& img) {
    CV_Assert(img.channels() == 1);
    cv::Mat gray;
    if (img.type() != CV_8U) {
        //img.convertTo(gray, CV_8U);
        cv::normalize(img, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    else gray = img;
    cv::imshow("normalized", gray);
    cv::waitKey(0);
    cv::destroyWindow("normalized");
    };

//C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23 GrayLut for Desktop at FH
//C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-22 GrayLut for Deskotp BMZ

auto show1to1 = [](const cv::Mat& img) {
    CV_Assert(img.channels() == 1);

    cv::Mat u8;
    if (img.type() == CV_8U) u8 = img;
    else cv::normalize(img, u8, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::namedWindow("normalized", cv::WINDOW_AUTOSIZE); // wichtig: 1:1
    cv::imshow("normalized", u8);
    cv::waitKey(0);
    cv::destroyWindow("normalized");
    };



int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2026-01-09Mako-WithGray4Shift" };

    std::string path_gray_sections{ "C:/Users/grein/Desktop/2026-01-19_GrayCalibSections2.csv" };
    //Path to IDS calibration: Blende geschlossen. 
    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2026-01-09MAKO.xml" };

    Deflectometry meassure{};

    std::vector<cv::Mat> real_pattern = 
        meassure.createSyntheticalImages();

    for (const auto& img : real_pattern) {
        showNormalized(img);
    }

    meassure.calc_response_curve_sections(1, 1, true, path_gray_sections, 8, 6);
    

    //meassure.do_camera_calibration();

    //meassure.do_grayvalue_calibration(5, "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2026-01-09BMZGrayValMako");


    // White balance
    /* std::vector<cv::Mat> frames =
        meassure.getFrames(FrameRole::Debug);

    std::array<double, 3> gain = meassure.doWhiteBalance(frames);*/

    //meassure.saveSingleImage(FrameRole::Debug, frames[0], true, "C:/Users/grein/Desktop/Master");
    //std::cout << "Gain values ";
    //for (const auto& val : gain) {
    //    std::cout << val << '\n';
    //}
    
    /*meassure.load(FrameRole::PatternDouble, path);
    std::vector<cv::Mat> pattern = meassure.get(FrameRole::PatternDouble);*/
    
    /*meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrast = meassure.get(FrameRole::Contrast);

    meassure.load(FrameRole::ReprojectionY, path);
    std::vector<cv::Mat> repro = meassure.get(FrameRole::ReprojectionY);

    cv::Mat mask000 = meassure.getMask(contrast, 0.3, true);

    cv::Mat subtracted =
        meassure.subtractSurface(contrast[0], mask000, true);*/

    std::vector<cv::Mat> raw_input =
        meassure.do_phase_measurement(Shift_mode::four_phase_shift, 5, true, path, 10);

    std::vector<cv::Mat> wrappedPhase =
        meassure.do_wrapped_phase(raw_input, 5, 4, true, path);
    
    meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);

    cv::Mat mask = meassure.getMask(contrastPhase, 0.3, false);

    std::vector<cv::Mat> unwrappedPhase =
        meassure.do_unwrapped_phase(wrappedPhase, mask, UnwrapMode::manually, true, path);

    cv::Mat reference_img = meassure.generate_reference_Pattern(ReferenceMode::checkerboard, (int)4, true, path);

    std::vector<cv::Mat> reference =
        meassure.getFrames(FrameRole::Debug, reference_img);

    mask = meassure.getMask(contrastPhase, 0.3, true);

    std::vector<cv::Vec2d> ref_point =
        meassure.getReferencePoint(reference, ReferenceMode::checkerboard, mask);


    // Method for projecting the reference pattern and getting camera images.
    /*std::vector<cv::Mat> cam_pattern 
        meassure.getFrames(FrameRole::all, pattern);*/

    //show1to1(pattern);

    //cv::Mat mask;// (pattern.size(), CV_8U);
    //std::vector<cv::Mat> pattern_vec{ pattern };

    std::cout << ref_point.size() << " Reference Point found. \n";
    for (const auto& point : ref_point) {
        std::cout << point << '\n';
    }

    /*meassure.load(FrameRole::RawPhase, path);
    std::vector<cv::Mat> unwrappedPhase = meassure.get(FrameRole::RawPhase);*/

    
    
    
    //// Calibrated Gray Value 4 Phase Shift - 
   /* meassure.load(FrameRole::UnwrappedPhase, path);
    std::vector<cv::Mat> unwrappedPhase = meassure.get(FrameRole::UnwrappedPhase);*/

    /*double min1, max1;
    cv::Point minLoc;

    cv::minMaxLoc(unwrappedPhase[0], &min1, &max1, &minLoc);
    std::cout << "min " << min1 << '\n' << "max " << max1 << '\n';

    cv::drawMarker(unwrappedPhase[0], minLoc, cv::Scalar(111), 0, 20);
    showNormalized(unwrappedPhase[0]);*/

    //cv::Mat cam_Matrix = cv::Mat::eye(3, 3, CV_64F);
    //cam_Matrix *= 2320;
    //cam_Matrix.at<double>(0, 2) = unwrappedPhase[0].cols / 2.0;
    //cam_Matrix.at<double>(1, 2) = unwrappedPhase[0].rows / 2.0;
    //cam_Matrix.at<double>(2, 2) = 1;

    ////cv::Size imageSize = unwrappedPhase[0].size();
    //std::vector<double> dist_coeff{ 0.094484605499573868 , 0.50993684205766665 , 0, -0, 0.13234471663910974 }; //     -0.0, 0.0, 0.0 , 0.0, 0.0
    //cv::Mat dist_coeffs(dist_coeff, true);


    /*std::cout << "Matrix: " << cam_Matrix << '\n';*/

    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> cam_Matrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> dist_coeffs = meassure.get(FrameRole::DistortionCoeff);

    /*std::vector<cv::Mat> cartesian = 
        meassure.generateCartesian(true, path, 1080/100, 1920/100);*/

    /*cv::Mat coordinateimg = meassure.generateCoordinateImg(true, path);*/

    
    /*meassure.load(FrameRole::WrappedPhase, path);
    std::vector<cv::Mat> wrappedPhase = meassure.get(FrameRole::WrappedPhase);*/


    /*std::vector<cv::Vec2d> startPts;

    for (std::size_t x = 0; x < 1920; ++x) {
        for (std::size_t y = 0; y < 1080; ++y) {
            startPts.emplace_back(cv::Vec2d(x, y));
        }
    }*/
   
    //meassure.distortionPipelineTest(cam_Matrix, dist_coeffs, startPts);

    /*std::vector<cv::Mat> pattern_distorted;
    for (auto& img : unwrappedPhase) {
        pattern_distorted.emplace_back(meassure.distortImage_manual(img, cam_Matrix, dist_coeffs));
    }*/
    
 /* std::vector<cv::Mat> undistorted1;
    for (auto& img : pattern_distorted) {
        undistorted1.emplace_back(meassure.undistortImage(img, cam_Matrix, dist_coeffs));
    }

    std::vector<cv::Mat> undistorted2;
    for (auto& img : pattern_distorted) {
        undistorted2.emplace_back(meassure.undistortImageManuell(img, cam_Matrix, dist_coeffs));
    }

    cv::Mat distortionError1 =
        meassure.calcDistortionError(undistorted1[0]);

    cv::Mat distortionError2 =
        meassure.calcDistortionError(undistorted2[0]);

    meassure.saveSingleImage(FrameRole::DistortErrX, distortionError1, true, path);

    meassure.saveSingleImage(FrameRole::DistortErrX, distortionError2, true, path);*/

    //dist_coeffs[0] = cv::Mat(5, 1, CV_64FC1, cv::Scalar(0));

    mask = meassure.getMask(contrastPhase, 0.3, true);

    std::vector<cv::Mat> reprojection = meassure.do_reprojection(
        unwrappedPhase,
        mask,
        ref_point[0],
        cam_Matrix[0],
        dist_coeffs[0],
        108.0,
        unwrappedPhase[0].cols,
        unwrappedPhase[0].rows,
        0.2745, //PixelPitch  FH 0.277    BMZ: 
        true,
        path,
        532, //Dispaly Width  FH    BMZ: 527.04
        299.2 //Dispaly Height FH:      BMZ: 296.46
    );

    /*cv::Mat distortionError =
        meassure.calcDistortionError(distorted);

    cv::Mat undistortionError =
        meassure.calcDistortionError(undistorted);

    meassure.saveSingleImage(FrameRole::DistortErrX, distortionError, true, path);

    meassure.saveSingleImage(FrameRole::DistortErrX, undistortionError, true, path);*/


    //meassure.loadPhaseConfig("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23/0.xml");

}
