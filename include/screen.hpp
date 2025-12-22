#ifndef SCREEN_H
#define SCREEN_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <vector>
#include <memory>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/calib3d.hpp>
//#include <opencv2/structured_light.hpp>
//#include <opencv2/phase_unwrapping.hpp>
#include "enums.hpp"
#include <array>
#include <optional>
#include "config/PhaseShiftConfig.hpp"


class ImageStore;

class Pattern {
public:
	// Constructor Pattern class (y_pixel, x_pixel, std::shared_ptr<ImageStore>)
	explicit Pattern(int y_pixel, int x_pixel, std::shared_ptr<ImageStore> m_img_store);

	enum class shift_axis {
		horizontal,
		vertikal,
		max_parameter
	};

	cv::Mat generateCross(
		const int pixel_x,
		const int pixel_y,
		const double mid_x,
		const double mid_y,
		const int thickness
	);

	cv::Mat generateCheckerboard(
		const int pixel_x,
		const int pixel_y,
		const int n_checker_size
	);

	//void grayValueCalibration();
	void generate_phaseShift(Shift_mode, int n_periods_y = 10);

	std::vector<cv::Mat> generateGrayCalibrationSequence(int stepwidth);

	cv::Mat generateCartesian(int gridX, int gridY);
	
	cv::Mat generatecoordianteImg();

	cv::Mat createCoordinateImg(int pixel_x, int pixel_y);

	/*void displayPatterns_single_thread();
	void displayPatterns_multi_thread();*/

	/*void gray_value_calib();*/

	void prepareLUT();

	void load_gray_calib_data(std::vector<std::pair<double, double>>&& lut) {
		m_LUT.emplace(std::move(lut));
		prepareLUT();
	}

	void generate_optimalPhase();

	std::vector<cv::Vec2i> getCartesianGridpoints(
		const cv::Size& sz,
		const int gridPointsX,
		const int gridPointsY
	);

	cv::Mat createCartesian(
		const int pixely, 
		const int pixelx, 
		const int gridY, 
		const int gridX);

	std::vector<std::shared_ptr<defl::PhaseShiftConfig>> getPhaseConfig() { return m_cfg; }

private:
	// Config file to save the options used for the pattern
	std::vector<std::shared_ptr<defl::PhaseShiftConfig>> m_cfg;

	// file with the purpose of holding all images used in this pipeline
	std::shared_ptr<ImageStore> m_img_store;

	int m_steps{};
	bool prepareShiftParameters(int n_periods_in_y, int m_steps);
	std::vector<cv::Mat> generateSinusPatternFromPhase(const std::vector<cv::Mat>&, int steps,
		double amplitude = 255, double mean_value = 127.5);

	std::vector<cv::Mat> generatePhase(int pixel_x, int pixely, double wavelength);
	cv::Mat generateRowPhase(int pixel_x, double wave_length);
	cv::Mat generateColumnPhase(int pixel_y, double wave_length);

	void logging();

	double linear_gray(double);

	int m_pixel_x{};
	int m_pixel_y{};
	double m_mean_value;
	double m_amplitude;
	double m_wavelength{};
	double m_numberPeriods{};
	double m_shift_length{};


	//Look Up Table
	std::optional<std::vector<std::pair<double, double>>> m_LUT{};
	std::vector<std::pair<double, double>> m_sortedLUT{};
	bool m_lut_ready{ false };
	double m_lut_scale_factor{};
	double m_lut_offset{};
};




#endif // !SCREEN_H
