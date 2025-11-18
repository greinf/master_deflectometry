#define VERSION "1"

#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>
#include "acquisitionworker.hpp"
#include <filesystem>
//#include "imgProcessing.hpp"

int main()
{
    
    //Supress open CV Information -only warnings are logged. 
    
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);


    Deflectometry meassure{};


    /*Does bind in a linearisation function in Screen class*/
    /*meassure.load_gray_value_calib("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-12_EX58773.5-FR 16.9987/Responsce20perFull.csv");
    meassure.gray_value_apply();*/

    
   /* meassure.grayValueCalib();
    std::string response_curve{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-12_EX58773.5-FR 16.9987/Responsce20perFull_calibrated.csv" };
    meassure.saveResponseCurve(response_curve);*/

    //Load calib does load the camera calibration data
    /*meassure.load_calib();
    meassure.start_meassurement(Shift_mode::four_phase_shift, DisplayMode::Automatic, 0);*/

    //// For debugging 
    meassure.TestOptimal();

    std::string frame_location("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-15_EX249965_FR3.99968NotCalibrated4Shift_100perPhase/Data.xml");

    //meassure.load_frames(frame_location);

    //OpenCV PhaseUnwrap
    //meassure.phase_unwrap(false, "C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out\\2025-07-11_Exposure_min*8000Calibrated");
   
    //Unwrap programmed myself
    //meassure.manual_phaseUnwrap();

    //std::string save_files{ "C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out\\2025-11-16_TestFR4VertikalError100FramesPhaseCalibrated" };
    //
    //meassure.calc_reproject_error(true, true,
    //   save_files);
    //
    //meassure.extract_Column(save_files + "\\Column.csv");
    //meassure.extract_Line(save_files + "\\Row.csv");

    /*
    std::string str("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/out_camera_data_First_real_calib.xml");
    calibrationData data(getfromFile(str));
    std::cout << "Camera Matrix: " << data.cameraMatrix << '\n' <<
        "Distortion Coefficients " << data.distCoeffs << std::endl;

        */

    /*
    //Iterating through a Folder with the openCv example pictures.
    std::vector<cv::Mat> vector{};
	std::string settings_path("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\data\\in_VID5.xml");
    std::string images("C:\\Users\\grein\\Desktop\\OpenCV_Example_CheckboardImages");
	std::filesystem::path image_files(images), settings_files(settings_path); 
    if(!std::filesystem::exists(image_files)) {
		std::cerr << "Settings file does not exist: " << image_files << '\n';
	}
    std::filesystem::directory_iterator dir_iter(image_files);
    for (const auto& entry : dir_iter) {
		std::cout << "Dir entry: " << entry.path() << '\n'; 
		vector.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE).clone());
    }
    
    */


    // This code can be used for calculating the backpropagation error. 
    //path to stored images
    /*
    std::filesystem::path file("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-10-23/2025-10-24CompleteData.xml");
    if (!std::filesystem::exists(file.parent_path())) {
        std::cerr << "Wrong addres used \n";
        return 0;
    }
    meassure.load_frames(file.string());
    */
}
