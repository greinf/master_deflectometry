#ifndef SCREEN_H
#define SCREEN_H

#include <atomic>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/phase_unwrapping.hpp>
#include "enums.hpp"
#include <array>

class Camera;
class AcquisitionWorker;

class Screen {
public:
	//Constructor (Width, Height, PixelPitch in micrometer, Number of periods 
	explicit Screen(std::int32_t screen_x = 1920, std::int32_t screen_y = 1080, std::int32_t pixel_pitch = 250, float m_numberperiods = (float)10);

	enum class shift_axis {
		horizontal,
		vertikal,
		max_parameter
	};

	//void grayValueCalibration();
	void generate_phaseShift(Shift_mode);
	
	//Displays Generated Pattern Sequence. Next Picture after 2sec. 
	void displayPatterns();
	std::vector<cv::Mat> m_patterns{};
	void showImage(const cv::Mat&);

	void stopDisplaying() {
		m_keepDisplaying.store(false);
	}

private:
	Shift_mode m_mode;
	int m_steps{};
	bool preparefourShiftParameters();
	bool generateSinusPatterns();

	std::atomic<bool> m_keepDisplaying{ true };
	std::int32_t m_pixel_x;
	std::int32_t m_pixel_y;
	std::int32_t m_pixel_pitch; 
	float m_amp{ 127.5 };
	float m_mean{ 127.5 };
	bool gray_val_calibrated{ false };
	float m_wavelength;
	float m_numberPeriods;
	float m_shift_length;

};




#endif // !SCREEN_H
