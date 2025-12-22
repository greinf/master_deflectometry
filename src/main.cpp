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
        img.convertTo(gray, CV_8U);
        cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    else gray = img;
    cv::imshow("normalized", gray);
    cv::waitKey(0);
    cv::destroyWindow("normalized");
    };

//C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23 GrayLut for Desktop at FH
//C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-22 GrayLut for Deskotp BMZ


int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    Deflectometry meassure{};
    //meassure.do_grayvalue_calibration(5,"C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-12-21NewGrayCalibDesktopFH" );

    //meassure.do_camera_calibration();

    /*std::vector<cv::Mat> frames = 
        meassure.getFrames(FrameRole::Debug);

    std::array<double, 3> gain = meassure.doWhiteBalance(frames);

    meassure.saveSingleImage(FrameRole::Debug, frames[0], true, "C:/Users/grein/Desktop/Master");
    std::cout << "Gain values ";
    for (const auto& val : gain) {
        std::cout << val << '\n';
    }*/

    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-12-22MeassureFullScreenFH" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-12_out_camera_data_First_real_calib.xml" };

    cv::Mat pattern = meassure.do_reference_Pattern(ReferenceMode::cross, (int)4, true, path);
    showNormalized(pattern);
    //std::vector<cv::Mat> pattern = meassure.generatePattern(true, path);
    
    //// Calibrated Gray Value 4 Phase Shift - 
    /*meassure.load(FrameRole::UnwrappedPhase, path);
    std::vector<cv::Mat> unwrappedPhase = meassure.get(FrameRole::UnwrappedPhase);*/

    /*double min1, max1;
    cv::Point minLoc;

    cv::minMaxLoc(unwrappedPhase[0], &min1, &max1, &minLoc);
    std::cout << "min " << min1 << '\n' << "max " << max1 << '\n';

    cv::drawMarker(unwrappedPhase[0], minLoc, cv::Scalar(111), 0, 20);
    showNormalized(unwrappedPhase[0]);*/

    //cv::Mat cam_Matrix = cv::Mat::eye(3, 3, CV_64F);
    //cam_Matrix *= 2320;
    //cam_Matrix.at<double>(0, 2) = unwrappedPhase[0].cols / 2.5;
    //cam_Matrix.at<double>(1, 2) = unwrappedPhase[0].rows / 2.5;
    //cam_Matrix.at<double>(2, 2) = 1;

    //cv::Size imageSize = unwrappedPhase[0].size();
    //std::vector<double> dist_coeff{ 0.094484605499573868 , 0.50993684205766665 , 0, -0, 0.13234471663910974 }; //-8.0E-8, 0.0, 0.0 , 0.0, 0.0
    //cv::Mat dist_coeffs(dist_coeff, true);

    /*std::cout << "Matrix: " << cam_Matrix << '\n';*/

   /* meassure.load(FrameRole::PatternDouble, path);
    std::vector<cv::Mat> pattern = meassure.get(FrameRole::PatternDouble);*/


    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> cam_Matrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> dist_coeffs = meassure.get(FrameRole::DistortionCoeff);

    /*std::vector<cv::Mat> cartesian = 
        meassure.generateCartesian(true, path, 1080/100, 1920/100);*/

    /*cv::Mat coordinateimg = meassure.generateCoordinateImg(true, path);*/

    /*std::vector<cv::Mat> rawPhase{ cartesian };*/



    std::vector<cv::Mat> raw_input =
        meassure.do_phase_measurement(Shift_mode::four_phase_shift, 5, true, path, 10);

    /*meassure.load(FrameRole::RawPhase, path);
    std::vector<cv::Mat> rawPhase = meassure.get(FrameRole::RawPhase);*/

    // Expect values between 0 - 111.64
    /*double min, max;

    cv::minMaxLoc(rawPhase[0], &min, &max);
    std::cout << "min " << min << '\n' << "max " << max << '\n';
    showNormalized(rawPhase[0]);*/
    /*std::vector<cv::Mat> pattern_distorted;
    for (auto& img : pattern) {
        pattern_distorted.emplace_back(meassure.distortImage_manual(img, cam_Matrix, dist_coeffs));
    }*/
    
    /*meassure.load(FrameRole::WrappedPhase, path);
    std::vector<cv::Mat> wrappedPhase = meassure.get(FrameRole::WrappedPhase);*/

    std::vector<cv::Mat> wrappedPhase =
        meassure.do_wrapped_phase(raw_input, 5, 4, true, path);

    meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);

    std::vector<cv::Mat> unwrappedPhase =
        meassure.do_unwrapped_phase(wrappedPhase, contrastPhase, UnwrapMode::manually, true, path);




    /*std::vector<cv::Vec2d> startPts;

    for (std::size_t x = 0; x < 1920; ++x) {
        for (std::size_t y = 0; y < 1080; ++y) {
            startPts.emplace_back(cv::Vec2d(x, y));
        }
    }*/
   
    //meassure.distortionPipelineTest(cam_Matrix, dist_coeffs, startPts);

    /*std::vector<cv::Mat> pattern_distorted;
    for (auto& img : rawPhase) {
        pattern_distorted.emplace_back(meassure.distortImage_manual(img, cam_Matrix, dist_coeffs));
    }
    
    std::vector<cv::Mat> undistorted1;
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

    //showNormalized(rawPhase[0]);
    //showNormalized(pattern_distorted[0]);

    //dist_coeffs[0] = cv::Mat(5, 1, CV_64FC1, cv::Scalar(0));

    std::vector<cv::Mat> reprojection = meassure.do_reprojection(
        unwrappedPhase,
        contrastPhase,
        cam_Matrix[0],
        dist_coeffs[0],
        108.0,
        unwrappedPhase[0].cols,
        unwrappedPhase[0].rows,
        0.277, //PixelPitch  0.2745
        true,
        path,
        532, //Dispaly Width 527.04
        299.2 //Dispaly Height 296.46
    );

    //meassure.saveSingleImage(FrameRole::DistortErrX, distorted, true, path);

   /* cv::Mat undistorted =
        meassure.undistortImage(img, cam_Matrix, dist_coeffs);

    cv::Mat undistortionError =
        meassure.calcDistortionError(undistorted);

    meassure.saveSingleImage(FrameRole::DistortErrX, undistortionError, true, path);

    cv::Mat distorted =
        meassure.distortImage_manual(undistorted, cam_Matrix, dist_coeffs);

    cv::Mat distortionError =
        meassure.calcDistortionError(distorted);

    meassure.saveSingleImage(FrameRole::DistortErrX, distortionError, true, path);*/

    /*cv::Mat distortionError =
        meassure.calcDistortionError(distorted);

    cv::Mat undistortionError =
        meassure.calcDistortionError(undistorted);

    meassure.saveSingleImage(FrameRole::DistortErrX, distortionError, true, path);

    meassure.saveSingleImage(FrameRole::DistortErrX, undistortionError, true, path);*/

   /* showNormalized2Channel(distorted);

    showNormalized2Channel(undistorted);*/


    

    
   
    

    //meassure.loadPhaseConfig("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23/0.xml");

    /*meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);*/

    /*meassure.load(FrameRole::UnwrappedPhase, path);
    std::vector<cv::Mat> unwrapped = meassure.get(FrameRole::UnwrappedPhase);*/
}



/*
std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23" };
    //std::vector<cv::Mat> raw_phase = 
    //    meassure.do_phase_measurement(Shift_mode::four_phase_shift, 5, true, path, 10);

    meassure.load(FrameRole::RawPhase, path);
    std::vector<cv::Mat> RawPhase = meassure.get(FrameRole::RawPhase);

    std::vector<cv::Mat> groundTruth;
    
    for (const auto& m : RawPhase) {
        int i{ 0 };
        cv::Mat out(m.rows, m.cols, CV_64F);

        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {

                double a = m.at<double>(r, c);
                double wrapped = std::fmod(a, CV_2PI);
                if (wrapped > CV_PI) wrapped -= CV_2PI;
                out.at<double>(r, c) = wrapped;
            }
        }
        groundTruth.push_back(out);
    }
    cv::Mat GT8U1;
    cv::Mat GT8U2;
    cv::normalize(groundTruth[0], GT8U1, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(groundTruth[1], GT8U2, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("GT1", GT8U1);
    cv::imshow("GT2", GT8U2);
    cv::waitKey(0);

    double min1, max1;
    cv::minMaxLoc(groundTruth[0], &min1, &max1);
    	std::cout << "Raw Ground Truth PHase \n" << "Minimal value: " << min1 <<
    		"\nMaxvalue: " << max1 << '\n';

    // Calibrated Gray Value 4 Phase Shift - 
    meassure.load(FrameRole::PatternDouble, path);
    std::vector<cv::Mat> gtpattern = meassure.get(FrameRole::PatternDouble);
    double min3, max3;
    cv::minMaxLoc(gtpattern[0], &min3, &max3);
    std::cout << "Raw ptPattern information \n" << "Minimal value: " << min3 <<
        "\nMaxvalue: " << max3 << '\n';

    std::vector<cv::Mat> wrappedPhase =
        meassure.do_wrapped_phase(gtpattern, 1, 4, false, path);

    double min2, max2;
    cv::minMaxLoc(wrappedPhase[0], &min2, &max2);
    std::cout << "Raw wrapped PHase \n" << "Minimal value: " << min2 <<
        "\nMaxvalue: " << max2 << '\n';
    
    cv::Mat unwrappedErr1, unwrappedErr2;

    unwrappedErr1 = wrappedPhase[0] - groundTruth[0];

    double min4, max4;
    cv::minMaxLoc(unwrappedErr1, &min4, &max4);
    std::cout << "Raw Ground Truth PHase \n" << "Minimal value: " << min4 <<
        "\nMaxvalue: " << max4 << '\n';


    for (int r = 0; r < unwrappedErr1.rows; ++r) {
        for (int c = 0; c < unwrappedErr1.cols; ++c) {

            double a = unwrappedErr1.at<double>(r, c);
            double wrapped = std::fmod(a, CV_2PI);
            if (wrapped > CV_PI) wrapped -= CV_2PI;

            unwrappedErr1.at<double>(r, c) = wrapped;
        }
    }

    double min5, max5;
    cv::minMaxLoc(unwrappedErr1, &min5, &max5);
    std::cout << "Error Ground Truth \n" << "Minimal value: " << min5 <<
        "\nMaxvalue: " << max5 << '\n';

    cv::Mat unwrappedErr1U8;
    cv::normalize(unwrappedErr1, unwrappedErr1U8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("ErrorU8", unwrappedErr1U8);
    cv::waitKey(0);

    cv::imwrite("C:/Users/grein/Desktop/UnwrapError1.png", unwrappedErr1U8);

    meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);

    std::vector<cv::Mat> unwrappedPhase =
        meassure.do_unwrapped_phase(wrappedPhase, contrastPhase, UnwrapMode::manually, true, path);

    meassure.load(FrameRole::RawPhase, path);
    std::vector<cv::Mat> rawPhase = meassure.get(FrameRole::RawPhase);

    cv::Mat error1, error2, error1U8, error2U8;
    error1 = unwrappedPhase[0] - rawPhase[0];
    error2 = unwrappedPhase[1] - rawPhase[1];

    double min7, max7;
    cv::minMaxLoc(error1, &min7, &max7);
    std::cout << "Raw Ground Truth PHase \n" << "Minimal value: " << min7 <<
        "\nMaxvalue: " << max7 << '\n';

    cv::normalize(error1, error1U8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(error2, error2U8, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imshow("er1", error1U8);
    cv::imshow("er2", error2U8);
    cv::waitKey(0);
    */