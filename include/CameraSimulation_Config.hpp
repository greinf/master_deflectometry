#ifndef CAMERASIMULATION_CONFIG_HPP
#define CAMERASIMULATION_CONFIG_HPP

#include <utils.hpp>
#include <vector>
#include <opencv2/core.hpp>


namespace cv {
	class Mat;
}

// Config file for the Camera Simulation Class
// All length are given in mm, and Angles in Grad
// CameraSimulationConfig:
// |-> CameraConfig:
// |	|-> bool quantization: If True applies Qunatization for the CameraIntensities
// |	|-> bool apertureSmoothing: If True a circular Kernel is used of the size of the circle of confusion.
// |	|		-> It is assumed that the Camera is focused on a point that is double the distance as the display 
// |	|-> int circle_of_confusion_n_disp: Specifies the circle of confusion in display pixels for the diameter of the used kernel.
// |	|		->if(circle_of_confusion_n_disp != 0 ) -> The kernel has diameter equal circle_of_confusion_n_disp
// |	|		->else -> The kernel size is calculated via distance, object size and sensor size parameters.
// |	|-> double f_number: The F-Number of the used objectiv of the camera. This is needed for the aperture smoothing
// |	|-> double sensor_size: The smaller sensor dimension is used. Example: 3,3" 8.8mm x 6.6mm -> use the smaller one
// |	|-> cv::Mat camera_mat: Camera Matrix from the camera calibration camera_mat.size() == cv::Size(3,3). camera_mat.type() == CV_64F
// |	|-> cv::Mat dist_coeffs: Dist Coeffs from the camera calibration. dist_coeffs.size() == cv::Size(1,n). camera_mat.type() == CV_64F
// |-> DisplayConfig:
// |	|-> bool displayQuantization: If true the display value are quantized to std::uint8_t
// |	|-> double pixelPitch: The display Pixlepitch
// |	|-> double gamma: The gamma value that is applied to the dispaly values. 
// |-> SceneConfig:
// |	|-> int disp_shift_z: The Distance in z_dir between disp and Camera coordinate System -> in the Camera System
// |	|-> int disp_shift_x: The Distance in x_dir between disp and Camera coordinate System -> in the Camera System
// |	|-> int disp_shift_y: The Distance in y_dir between disp and Camera coordinate System -> in the Camera System
// |	|-> int disp_tilt_x: The Angle around x-Axis between disp and Camera coordiante System -> from the view of the Camera System [rad]
// |	|-> int disp_tilt_y: The Angle around y-Axis between disp and Camera coordiante System -> from the view of the Camera System [rad]
// |	|-> double object_size: Dimension of the Object. If for x and y different -> take the bigger one. 
// |	|		-> Here a cicualr mirror is used with 400mm diameter. 
// |-> Data:
// |	|-> std::vector<cv::Mat>::const_iterator begin: an Iterator to the start position of the Images to use for simulation
// |	|-> std::vector<cv::Mat>::const_iterator end: an Iterator to end position of the Images to use for simulatio
// |	|-> std::vector<cv::Mat>*: an Ptr to a std::vector<cv::Mat> to store the output images to

struct CameraSimulationConfig {

private:
	struct CameraConfig {
		bool quantization{ true };
		bool apertureSmoothing{ false };
		int circle_of_confusion_n_disp{ 0 };
		double f_number{ 2.4 };
		double sensor_size{ 6.6 }; 
		cv::Mat camera_mat;
		cv::Mat dist_coeffs;
		int pixel_x{ 2464 };
		int pixel_y{ 2056 };
	};

	struct DisplayConfig {
		bool displayQuantization{ true };
		double pixelPitch{ 0.2745 };
		double gamma{ 1 };
	};

	// All translations and Rotations are in the camera coordiante System
	struct SceneConfig {
		double disp_shift_z{ 3000.0 };  // z-Dist from Cam coord to Disp coord. Defaulted to 3000 [mm]
		double disp_shift_x{ 0.2745 * 1920/2 }; // x-Dist from Cam coord. to Disp coord. Defaulted to 0.2745*1920/2 -> Display coord in left upper corner. [mm]
		double disp_shift_y{ 0.2745 * 1080/2 }; // y-Dist from Cam coord. to Disp coord. Defaulted to 0.2745*1080/2 -> Display coord in left upper corner. [mm]
		double disp_tilt_x{ }; 
		double disp_tilt_y{ };
		double object_size{400.0};
		bool luminance{false};
	};

	struct Data {
		// Data To work on. Ptrs to the images
		const cv::Mat* begin = nullptr;
		const cv::Mat* end = nullptr;
		// Ouptut Vector for the images
		std::vector<cv::Mat>* output = nullptr;
	};

public:
	CameraConfig camera{};
	DisplayConfig disp{};
	SceneConfig scene{};
	Data data{};
};



#endif