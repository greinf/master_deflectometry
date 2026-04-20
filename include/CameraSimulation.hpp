#ifndef CAMERASIMULATION_HPP
#define CAMERASIMULATION_HPP
#include "CameraSimulation_Config.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <memory>

class ImageProcessing;
class GrayCalibration;


class CameraSimulation {
private:
	ImageProcessing& m_img_processing;
	GrayCalibration* m_gray_calibration = nullptr;

	// Input[0]: CameraSimulationConfig&
	// | ....
	// |-> Data:
	// |	|-> std::vector<cv::Mat>::const_iterator begin.
	// |	|-> std::vector<cv::Mat>::const_iterator end.
	// A deep copy of [begin,end) is constructed and saved in m_impl data holder
	void extractImages(const CameraSimulationConfig&);

	std::vector<cv::Mat>& applyApertureSmoothing(
		const CameraSimulationConfig& config
		);

	std::vector<cv::Mat>& applyScalingAndBias(
		const CameraSimulationConfig& config
	);

	struct Impl;
	std::unique_ptr<Impl> m_impl = nullptr;

	cv::Mat createCircularBinaryMask(const int diameter);

	// Input[0]: const cv::Size& sz. width and height of image to create
	// Input[1]: const int dimensions. 
	//		-> Allowed Values: [2,3] 
	// Output:
	//		if 2: Output cv::Mat_<cv::Vec2d> with cv::Vec2d = (x ,y) per pixel
	//		if 3: Output cv::Mat_<cv::Vec3d> with cv::Vec3d = (x,y,0) per pixel
	cv::Mat generateCoordinateImage(
		const cv::Size& sz,
		const int dimension = 2
	);

	void mirrorRays(
		cv::Mat_<cv::Vec3d>& rays,
		const cv::Vec3d surface_normal
	);

	cv::Mat calcDisplayPointinCameracoordinates(
		const cv::Size& pattern_size,
		const double shift_z,
		const double shift_x,
		const double shift_y,
		const double tilt_x,
		const double tilt_y,
		const double pixel_pitch,
		bool flip_vertical = false
	);

	std::vector<cv::Mat> remapFromHitpoints(
		const std::vector<cv::Mat>::iterator start,
		const std::vector<cv::Mat>::iterator end,
		const cv::Mat_<cv::Vec2d>& hitpoints_camera_world
	);

public:
	CameraSimulation(ImageProcessing& img_processing);

	/*CameraSimulation(
		ImageProcessing& img_processing,
		GrayCalibration* calibration)
		:m_img_processing{img_processing}
		,m_gray_calibration{calibration}
	{ }*/

	~CameraSimulation();

	std::vector<cv::Mat> simulate(CameraSimulationConfig&);

	CameraSimulation(CameraSimulation&&) noexcept;
	CameraSimulation& operator=(CameraSimulation&&) noexcept = delete;

	CameraSimulation(const CameraSimulation&) = delete;
	CameraSimulation& operator=(const CameraSimulation&) = delete;
};





#endif