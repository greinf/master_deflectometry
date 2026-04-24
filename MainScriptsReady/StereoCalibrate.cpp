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

    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-11_Grauwertkalbirierung_Data_Simulated/2026-04-11_Sim_G2.2_nLumQuantDispApertureQantCam" };

    std::string camMatrix_save_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie"
        "/data/2026-04-23_CameraCalibration/Stereo_Calib.xml" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    CalibrationConfig CamCalibConfig;
    CamCalibConfig.fixK2 = true;
    CamCalibConfig.fixK3 = true;
    CamCalibConfig.fixK4 = true;
    CamCalibConfig.fixK5 = true;
    CamCalibConfig.boardSize = { 8, 11 };
    CamCalibConfig.squareSize = 30.0;
    CamCalibConfig.outputFileName = camMatrix_save_path;
    CamCalibConfig.writeExtrinsics = true;
    CamCalibConfig.writePerViewErrors = true;

    std::vector<cv::Mat> cam1, cam2;

    // Cam2 is primary 
    for (std::size_t i = 0; i < 12; ++i) {
        std::vector<cv::Mat> calib_frames1 = meassure.getFrames(FrameRole::Debug, 1);
        std::vector<cv::Mat> calib_frames2 = meassure.getFrames(FrameRole::Debug, 1);
        cam1.push_back(calib_frames1[0]);
        cam2.push_back(calib_frames2[0]);
    }

    std::vector<StereoImagePair> images;

    for (std::size_t i = 0; i < cam1.size(); ++i) {
        StereoImagePair imgpair;
        imgpair.right = cam1[i];
        imgpair.left = cam2[i];
        images.push_back(imgpair);
    }

    CameraCalibration calib{};

    auto result = calib.calibrateStereo(
        images,
        CamCalibConfig,
        true);

    std::cout << "do something. " << std::endl;


}


