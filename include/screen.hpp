#ifndef SCREEN_H
#define SCREEN_H


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

#include <array>

class Screen {
public:
	//Constructor (Width, Height, PixelPitch in micrometer, Number of periods 
	explicit Screen(std::int32_t screen_x = 1920, std::int32_t screen_y = 1080, std::int32_t pixel_pitch = 250, float m_numberperiods = (float)10);

	enum class Shift_mode {
		four_phase_shift,
		beat_frequency_shift,
		max_values
	};

	enum class shift_axis {
		horizontal,
		vertikal,
		max_parameter
	};

	std::array<std::string, static_cast<size_t>(Shift_mode::max_values)> shift_mode_string {"four phase shift ",
		"beat frequency shift "};

	//void grayValueCalibration();
	void generate_phaseShift(Shift_mode);
	//void generate_phaseShift();
	void displayPattern();

private:
	//bool preparePhaseshift(bool horizontal_shift, const Shift_mode mode = Shift_mode::four_phase_shift);
	bool preparefourShiftParameters();
	bool generateSinusPatterns();
	/*
	cv::structured_light::SinusoidalPattern::Params m_params;
	
	cv::Ptr<cv::structured_light::SinusoidalPattern> m_sinus_vertical = 
		cv::structured_light::SinusoidalPattern::create(cv::makePtr<cv::structured_light::SinusoidalPattern::Params>(m_params));

	cv::Ptr<cv::structured_light::SinusoidalPattern> m_sinus_horizontal = 
		cv::structured_light::SinusoidalPattern::create(cv::makePtr<cv::structured_light::SinusoidalPattern::Params>(m_params));
	*/
	Shift_mode m_mode;
	std::int32_t m_pixel_x;
	std::int32_t m_pixel_y;
	std::int32_t m_pixel_pitch; 
	float m_amp{ 127.5 };
	float m_mean{ 127.5 };
	bool gray_val_calibrated{ false };
	float m_wavelength;
	float m_numberPeriods;
	float m_shift_length;
	std::vector<cv::Mat> m_patterns{};
};




#endif // !SCREEN_H
