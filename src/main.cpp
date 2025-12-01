#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>
#include "acquisitionworker.hpp"
#include <filesystem>


//meassure.do_grayvalue_calibration(1);

//meassure.do_camera_calibration();

int main()
{
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    Deflectometry meassure{};

    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23" };

    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-12_out_camera_data_First_real_calib.xml" };

    //meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);

    /*std::vector<cv::Mat> raw_phase = 
        meassure.do_phase_measurement(Shift_mode::four_phase_shift, 5, true, path, 10);*/

    // Calibrated Gray Value 4 Phase Shift - 
    meassure.load(FrameRole::RawPhase, path);
    std::vector<cv::Mat> raw_input = meassure.get(FrameRole::RawPhase);

    /*std::vector<cv::Mat> wrappedPhase =
        meassure.do_wrapped_phase(raw_input, 5, 4, true, path);*/

    /*meassure.generatePattern(true, path);*/

    /*std::vector<cv::Mat> cam_Matrix = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> dist_coeffs = meassure.get(FrameRole::DistortionCoeff);*/

    /*meassure.load(FrameRole::PatternDouble, path);
    std::vector<cv::Mat> pattern = meassure.get(FrameRole::PatternDouble);*/

    meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);

    cv::Mat img = raw_input[0];

    cv::Mat cam_Matrix = cv::Mat::eye(3, 3, CV_64F);
    cam_Matrix *= 800;
    cam_Matrix.at<double>(0, 2) = img.cols / 2.0;
    cam_Matrix.at<double>(1, 2) = img.rows / 2.0;
    cam_Matrix.at<double>(2, 2) = 1;
   
    cv::Size imageSize = img.size();
    std::vector<double> dist_coeff{ 8.0E-3 , 0.00008, 0.0 , 0.0, 0.0 }; //0.0 , 0.0, 0.0 , 0.0, 0.0
    cv::Mat dist_coeffs(dist_coeff, true);

    std::cout << "Matrix: " << cam_Matrix << '\n';

    std::vector<cv::Mat> pattern_distorted;
    for (auto& img : raw_input) {
        pattern_distorted.emplace_back(meassure.distortImage_manual(img, cam_Matrix, dist_coeffs));
    }

    std::vector<cv::Mat> reprojection = meassure.do_reprojection(
        pattern_distorted,
        contrastPhase,
        cam_Matrix,
        dist_coeffs,
        108.0,
        raw_input[0].cols,
        raw_input[0].rows,
        0.2745, //PixelPitch
        true,
        path,
        527.04, //Dispaly Width
        296.46 //Dispaly Height
    );


    cv::Mat distroted = 
        meassure.distortImage_manual(img, cam_Matrix, dist_coeffs);


    cv::Mat undistorted =
        meassure.undistortImage(distroted, cam_Matrix, dist_coeffs);

    

    /*meassure.load(FrameRole::WrappedPhase, path);
    std::vector<cv::Mat> wrappedPhase = meassure.get(FrameRole::WrappedPhase);*/

    
   
    /*std::vector<cv::Mat> unwrappedPhase =
        meassure.do_unwrapped_phase(wrappedPhase, contrastPhase, UnwrapMode::manually, true, path);*/

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