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
    
    std::filesystem::path file("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-10-23/2025-10-24CompleteData.xml");
    if (!std::filesystem::exists(file.parent_path())) {
        std::cerr << "Wrong addres used \n";
        return 0;
    }
    

    //meassure.grayValueCalib();

    meassure.load_frames(file.string());
    meassure.load_calib();
    meassure.calc_reproject_error();

    //meassure.save_frames(file.string());

    
    //meassure.start_meassurement(Shift_mode::four_phase_shift, DisplayMode::Automatic, 0);

    //meassure.show_acquistion();
    //meassure.phase_unwrap();

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
    

	runCameraCalibration(vector, true, settings_files.string());
    */

    /*
    
    */
}
