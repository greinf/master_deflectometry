#ifndef CAMERASIMULATION_CONFIG_HPP
#define CAMERASIMULATION_CONFIG_HPP

#include <utils.hpp>

namespace cv {
	class Mat;
}

// Config file for the Camera Simulation Class
// All length are given in mm, and Angles in Grad
// CameraSimulationConfig:
// |->CameraConfig
// |	|-> bool quantization: If True applies Qunatization for the CameraIntensities
// |	|-> bool apertureSmoothing: If True a circular Kernel is used of the size of the circle of confusion.
//									-> It is assumed that the Camera is focused on a point that is double the distance as the display 
// |	|-> double sensor
//
struct CameraSimulationConfig {
private:
	enum warping {
		homography,
		raycasting
	};

	struct CameraConfig {
		bool quantization{ true };
		bool apertureSmoothing{ true };
		double f_number{ 2.4 };
		double sensor{ 6.6 };
		cv::Mat camera_mat;
		cv::Mat dist_coeffs;
	};

	struct DisplayConfig {
		bool displayQuantization{ true };
		double pixelPitch{ 0.2745 };
		double gamma{ 1 };
	};

	// All translations and Rotations are in the camera coordiante System
	struct SceneConfig {
		int disp_shift_z{ 0 };  // As the distance is calcualted in camera Coords -> Z is the distance in the direction of the "Rays"
		int disp_shift_x{ 0 };
		int disp_shift_y{ 0 };
		int disp_tilt_x{ 0 };
		int disp_tilt_y{ 0 };
		warping warping{ raycasting };
		RoiBorders<double> border;
	};

	struct Data {
		std::vector<cv::Mat>::const_iterator begin;
		std::vector<cv::Mat>::const_iterator end;
	};

public:
	CameraConfig camera{};
	DisplayConfig disp{};
	SceneConfig scene{};

};



#endif