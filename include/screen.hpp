#ifndef SCREEN_H
#define SCREEN_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <vector>
#include <memory>

#include "enums.hpp"
#include <array>
#include <optional>
#include "config/PhaseShiftConfig.hpp"
#include "RowPolicy.hpp"

class ImageStore;

namespace uniform {
	constexpr bool is_uniform_rows(UniformRowsCols) { return true; }
	constexpr bool is_uniform_rows(PerElement) { return false; }
}

class Pattern {
public:
	// Constructor Pattern class (y_pixel, x_pixel, std::shared_ptr<ImageStore>)
	explicit Pattern(ImageStore& img_store);

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

	// Row Policy can speed up computation if the values only have to be calculate for one row or column. 
	template<typename RowPolicy = UniformRowsCols>
	std::vector<cv::Mat> generate_phaseShift(Shift_mode mode = Shift_mode::four_phase_shift,
		int n_periods_y = 10,
		double mean = 127.5,
		double ampl = 127.5,
		int pixelX = 1920,
		int pixelY = 1080,
		RowPolicy policy = {})
	{
		CV_Assert(n_periods_y >= 1);
		CV_Assert(pixelX >= 1);
		CV_Assert(pixelY >= 1);
		const bool uniformRow = uniform::is_uniform_rows(policy);
		return generate_phaseShift(mode, n_periods_y, mean, ampl, pixelX, pixelY, uniformRow);
	}

	// Creates a 
	std::vector<cv::Mat> generateGrayCalibrationSequence(
		const int stepwidth,
		const int pixel_x = 1920,
		const int pixel_y = 1080);

	cv::Mat generateCartesian(int gridX, int gridY);

	// Function creates a Picutre of cv::size(pixel_x, pixel_y) with channels()  = 2
	// Per Pixel the Coords are saved (x,y)
	cv::Mat generatecoordianteImg(
		const int pixel_x = 1920,
		const int pixel_y = 1080);

	// Function creates a Picutre of cv::size(pixel_x, pixel_y) with channels()  = 2
	// Per Pixel the Coords are saved (x,y)
	cv::Mat createCoordinateImg(int pixel_x, int pixel_y);

	void generate_optimalPhase();

	// Input:
	// 1 cv::size() of image
	// 2 ammount of equally spaced gridPoints in x direction
	// 3 ammound of equally spaced gridPoints in y direction
	// Returns: a vector of cv::Vec2i wirh coordinates for marker. 
	std::vector<cv::Vec2i> getCartesianGridpoints(
		const cv::Size& sz,
		const int gridPointsX,
		const int gridPointsY
	);

	// Input:
	// 1 pixel_y (pixel in y direction)
	// 2 pixel_x (pixel in x direction)
	// 3 gridY - ammount of equally spaced markers in y direction
	// 4 gridX - ammount of equally spaced markers in x direction
	cv::Mat createCartesian(
		const int pixely, 
		const int pixelx, 
		const int gridY, 
		const int gridX);

	std::vector<defl::PhaseShiftConfig> getPhaseConfig() const;

private:
	// Config file to save the options used for the pattern
	std::vector<std::shared_ptr<defl::PhaseShiftConfig>> m_cfg;
	int m_steps{};

	// file with the purpose of holding all images used in this pipeline
	ImageStore& m_img_store;

	std::vector<cv::Mat> generate_phaseShift(
		Shift_mode,
		int n_periods_y = 10,
		double mean = 127.5,
		double ampl = 127.5,
		int pixelX = 1920,
		int pixelY = 1080,
		bool uniformRow = true);

	bool prepareShiftParameters(
		int n_periods_in_y, 
		int m_steps);

	// This method expects a std::vector<cv::Mat> of size() =2
	// phasemap[0]: HorizontalPhase
	// phasemap[1]: VerticalPhase
	std::vector<cv::Mat> generateSinusPatternFromPhase(
		const std::vector<cv::Mat>& phase_map, 
		int steps,
		double amplitude = 127.5, 
		double mean_value = 127.5,
		bool uniformRow = true);

	std::vector<cv::Mat> generatePhase(
		int pixel_x, 
		int pixely, 
		double wavelength);

	cv::Mat generateRowPhase(
		int pixel_x, 
		double wave_length);

	cv::Mat generateColumnPhase(
		int pixel_y, 
		double wave_length);

	void logging();

	

	int m_pixel_x{};
	int m_pixel_y{};
	double m_mean_value{};
	double m_amplitude{};
	double m_wavelength{};
	double m_numberPeriods{};
	double m_shift_length{};

};




#endif // !SCREEN_H
