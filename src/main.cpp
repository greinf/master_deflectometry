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
    std::filesystem::path inFolder = R"(C:\Users\grein\Desktop\Data)";
    std::filesystem::path outFolder = R"(C:\Users\grein\Desktop\Data_outBigDistance)";
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/1Wave" };

    std::string path_gray_sections{ "C:/Users/grein/Desktop/2026-02-15_GrayCalibSections2.csv" };

    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-15_GrayCalibSections" };
    //Path to IDS calibration: Blende geschlossen. 
    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-09_StereoMAKO.xml" };

    std::string camMatrix_path1{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-09_StereoMAKO" };


    std::string calibPath{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2026-02-09StereoCalibration" };

    std::string calibrationDisplayCampath{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-17_Display_Cam" };

    Deflectometry meassure{};
    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix_test_simulation = meassure.get(FrameRole::CalibrationMatrix);
    std::vector<cv::Mat> distCoeffs = meassure.get(FrameRole::DistortionCoeff);

    GrayCodeConfig config{};
    config.creation.inverse = false;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 1920;
    config.creation.resolution_y = 1080;
    config.creation.starBit = config.msb;
    
    std::vector<cv::Mat> grayCode1 = pat.generateGrayCodeImg(config);
    
    CameraSimulation simulation(processing);

    CameraSimulationConfig Sim_config;
    Sim_config.camera.camera_mat = camMatrix_test_simulation[0];
    Sim_config.camera.dist_coeffs = distCoeffs[0];
    Sim_config.scene.disp_shift_z = 4000.0;
    Sim_config.scene.disp_shift_x = -1920 /2.0 * 0.2745;
    Sim_config.scene.disp_shift_y = -1080 / 2.0 * 0.2745;
        
    //Sim_config.scene.disp_tilt_x = CV_PI / 10.0;
    Sim_config.scene.disp_tilt_y = CV_PI / 10.0;
    Sim_config.disp.gamma = 2.2;
    Sim_config.data.begin = grayCode1.begin()._Ptr;
    Sim_config.data.end = grayCode1.end()._Ptr;
   
    std::vector<cv::Mat> simulate = 
        simulation.simulate(Sim_config);

    auto& eval = config.getEvalParameter();
    eval.start = simulate.begin();
    eval.end = simulate.end();

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    std::vector<cv::Mat> result{ config };
    
    cv::Mat homography = 
        processing.createHomographyFromGrayCode(config, cv::Size(1920, 1080));

    std::vector<cv::Mat> maps = processing.createMappingfromHomography(homography, cv::Size(1920, 1080));

    std::vector<cv::Mat> grayCodemapped = processing.remapCameraToScreen(simulate, maps);

    meassure.GrayCalibrationClassTest(_defl_::GrayCal::Method::ActiveLut);

    // AbstandsMessung 
    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);

    meassure.load(FrameRole::CalibCamToCam, camMatrix_path);
    std::vector<cv::Mat> camToCam = meassure.get(FrameRole::CalibCamToCam);

    std::vector<cv::Mat> dist_Coeffs = meassure.get(FrameRole::DistortionCoeff);

    std::optional<std::vector<cv::Mat>> cameramatrix_distCoeff;
    cameramatrix_distCoeff.emplace({ camMatrix[0], dist_Coeffs[0] });

    meassure.testReprojection(
        camMatrix_path,
        path_gray_calibration,
        path,
        Shift_mode::four_phase_shift,
        _defl_::GrayCal::Method::ActiveLut,
        33.75,
        false,
        UnwrapMode::opencv,
        true);

    
    //// ************ Triangulation *******************

    //auto result = meassure.computeLaserDistanceFrom4Images(
    //    LaserFr[0], LaserFr[1], LaserFr[2], LaserFr[3],
    //    camMatrix[0], dist_Coeffs[0], camMatrix[1], dist_Coeffs[1], camToCam[0], camToCam[1]
    //);

    //std::cout << "p1 (px): " << result.p1_px << "\n";
    //std::cout << "p2 (px): " << result.p2_px << "\n";
    //std::cout << "3D (cam1): [" << result.X_cam1.x << ", " << result.X_cam1.y << ", " << result.X_cam1.z << "]\n";
    //std::cout << "distance to cam1: " << result.distance_cam1 << " (same units as T)\n";


    double pixelpitch = 0.2745;

    std::vector<cv::Mat> pattern10 =
        meassure.generatePattern(Shift_mode::four_phase_shift, _defl_::GrayCal::Method::None, path_gray_calibration, FrameRole::Debug, false, " ", 10);

    
    std::vector<cv::Mat> wrappedPhase =
        meassure.do_wrapped_phase(pattern10, 1, 4, true, path);

    //meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);

    cv::Mat mask = meassure.getMask(contrastPhase, 0.0, false);

    std::vector<cv::Mat> unwrappedPhase =
        meassure.do_unwrapped_phase(wrappedPhase, mask, UnwrapMode::manually, true, path, 108);

    cv::Mat reference_img =
        meassure.generate_reference_Pattern(ReferenceMode::checkerboard, (int)4, true, path);

   /* std::vector<cv::Mat> reference =
        meassure.getFrames(FrameRole::Debug, 1, reference_img);*/

    mask = meassure.getMask(contrastPhase, 0.0, true);

    std::vector<cv::Vec2d> ref_point =
        meassure.getReferencePoint(
            std::vector<cv::Mat>{reference_img},
            ReferenceMode::checkerboard, 
            mask);

    std::vector<cv::Mat> pattern_distorted;
    for (auto& img : unwrappedPhase) {
        pattern_distorted.emplace_back(meassure.distortImage_manual(img, camMatrix[0], dist_Coeffs[0]));
    }


    std::vector<cv::Mat> reprojection = meassure.do_reprojection(
        pattern_distorted,
        mask,
        ref_point[0],
        camMatrix[0],
        dist_Coeffs[0],
        108.0,
        unwrappedPhase[0].cols,
        unwrappedPhase[0].rows,
        0.2745, //PixelPitch  FH 0.277    BMZ: 
        true,
        path,
        532, //Dispaly Width  FH    BMZ: 527.04
        299.2 //Dispaly Height FH:      BMZ: 296.46
    );


    mask = meassure.getMask(contrastPhase, 0.3, true);

    std::vector<cv::Mat> reprojection1 = meassure.do_reprojection(
        unwrappedPhase,
        mask,
        ref_point[0],
        camMatrix[0],
        dist_Coeffs[0],
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
