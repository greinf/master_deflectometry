#ifndef CAMERASIMULATION_HPP
#define CAMERASIMULATION_HPP
#include "CameraSimulation_Config.hpp"

class ImageProcessin;
class GrayCalibration;


class CameraSimulation {
	

	ImageProcessin& m_img_processing;
	GrayCalibration* m_calibration = nullptr;


public:
	CameraSimulation(ImageProcessing& img_processing) :
		m_img_processing{ img_processing } 
	{ }
	CameraSimulation(
		ImageProcessin& img_processing,
		GrayCalibration* calibration)
		:m_img_processing{img_processing}
		,m_calibration{calibration}
	{ }






};





#endif