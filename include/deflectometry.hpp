#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H

#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>
#include <opencv2/opencv.hpp>


//Forward Decleration Enums + Class
enum class AcquisitionMode;
enum class Shift_mode;
enum class DisplayMode;

class Screen;
class AcquisitionWorker;
class ImageProcessing;

class Deflectometry {
public:
	 Deflectometry();

	 void start_meassurement(Shift_mode, DisplayMode, int);
	 void show_acquistion();
	 void phase_unwrap();

	 void camera_calibration(int camera);
	 
	 // Access Image Processing class, there all imgaes of the PhaseUnwrap are stored. 
	 // A .xml file is created at the given address, where the image data can be accesed. The
	 // data is stored in the original CV_32F format. Therefore not showable.
	 // Acces image files by creating a cv::FileStorage fs Instance at this point. and fs["std::string"] >> cv::Mat 
	 void save_frames(std::string&);

	 // Stores 8 bit images in .png / .jpg format. Images have to be given in in a uchar 8 bit with 1 or 3 channels, in 
	 // a vector<cv::Mat> format and the address where the files need to be stored. 
	 // If floating point images are given to this function, cv::imwrite will try to save them which leads to data loss. 
	 void save_frames(std::vector<cv::Mat>& frames, const std::string& path);

	 void calc_reproject_error(bool visualizing = true, bool saving = false, const std::string& path = "");

	 void load_frames(const std::string& path);
	 void generatePattern();
	 void load_calib(std::string path = "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-10-23/2025_10-23_Camera_calib.xml");

	 void grayValueCalib(int camera = 0);

	 void saveResponseCurve(const std::string& filename);

private:
	std::shared_ptr<Screen> m_screen{nullptr};
	std::shared_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };
	std::shared_ptr<ImageProcessing> m_img_processing{ nullptr };
	std::thread img_handler_thread;

	// Contoller_thread that is used to oversee the acquisition of camera frames
	// druing the meassurment. This is done by UserInput.
	void controller_userInput();


	
	// Controller thread for automatic acquisaition. Default argument is ammount of pictures taken per 
	// Meassurment. Information about ammount of shift_steps is saved in flagHandler.hpp.
	void controller_automatic();
	void controller_automatic_gray(std::unique_lock<std::mutex>&& lk_pattern, std::unique_lock<std::mutex>&& lk_save);
	
};

#endif 