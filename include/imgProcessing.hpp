#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include "flagHandler.hpp"
#include "imageHandler.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <tuple>

struct calibrationData {
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
};

inline calibrationData getfromFile(std::string& path) {
	cv::FileStorage fs(path, cv::FileStorage::READ);
	calibrationData data;
	fs["distortion_coefficients"] >> data.distCoeffs;
	fs["camera_matrix"] >> data.cameraMatrix;
	return data;
}

class ImageProcessing {
public:
	ImageProcessing();

	void gray_value_calib(std::vector<cv::Mat>&);

	void wrapped_phase();

	cv::Mat createMask();

	void bayerToGray();

	void unwrapped_phase();

	void goldsteinUnwrap();

	// This function saves uncompressed images to a .xml file. This allows for storing of floating point values. 
	void saveImages(std::string& path);

	void load_frames(const std::string& path);

	void load_calib(std::string path);

	void calc_reproject_error(bool visualizing);

	std::array<cv::Mat, (std::size_t) 2> m_baseIntensity;
	std::array<cv::Mat, (std::size_t) 2> m_contrast;
	std::array<cv::Mat, (std::size_t) 2> m_phase;

	std::vector<cv::Scalar_<double>>& get_mean_values() {
		return m_mean_grayValues;
	}

	//void convertToFloat(std::vector);
	~ImageProcessing();
	ImageProcessing(const ImageProcessing&) = delete;
	ImageProcessing& operator=(const ImageProcessing&) = delete;
	ImageProcessing(ImageProcessing&&) = delete;
	ImageProcessing& operator=(ImageProcessing&&) = delete;

	void assginFrames(std::vector<cv::Mat>&& mat) {
		m_frames = mat;
	}
	
	template<typename func, typename... Mats>
	static auto forEachPixel(func, Mats&&... mats);

	struct minmaxloc {
		double minval{}, maxval{};
		cv::Point minloc{}, maxloc{};

		friend std::ostream& operator<<(std::ostream& out, minmaxloc loc) {
			std::cout << "Minimum value : " << loc.minval << " at location: " << loc.minloc << '\n' <<
				"Maximum value : " << loc.maxval << "at location: " << loc.maxloc << '\n';
			return out;
		}
	};

	[[maybe_unused]] std::vector<cv::Mat> m_reprojection_error_img;
private:	
	// If number_of_frames_per_pattern > 0. Here mean values get stored. 
	// Starting point for Image Processing steps. 
	static inline int instance_counter{ 0 };
	std::vector<cv::Mat> m_frames{};
	std::vector<cv::Mat> m_raw_phase{};
	
	cv::Mat m_mask;
	
	std::array<cv::Mat, (std::size_t)2> m_unwrapped_phase{};
	std::array<cv::Mat, (std::size_t)2> m_wrapped_phase{};
	cv::Mat mean(std::vector<cv::Mat>&);

	std::array<cv::Mat, (std::size_t)2> m_s1{};
	std::array<cv::Mat, (std::size_t)2> m_s2{};
	std::array<cv::Mat, (std::size_t)2> m_s3{};
	
	std::string path{ "C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out" };

	std::vector<cv::Scalar_<double>> m_mean_grayValues;
	// Initializiation of empty cv::Mats in the right datatype. This is necessary for the calculation of the wrapped phase. 
	void create();
	void create(std::vector<cv::Mat>& vec);

	// First try to write a tempalte function that allows to input a arbitrary number of input matrices
	// The template iterates through all arrays simultaniously and allows to operato on the elemnts. 
	
	
	template<int Depth, typename Func, typename... Mats>
	static cv::Mat forEachPixelImpl(const Func& func, Mats&&... mats);

	cv::Mat applyMask(const cv::Mat&, const cv::Mat&, float shift = 0.0f);

	//Loads images ín png or jpeg format, from the given path.
	cv::Mat load_images(std::string path_to_image);
	
	// Just have some fun with std::reference_wrapper. Extremly good c++ documentation code for implementation of generic std::reference_wrapper. 
	// std::string str{ "roflcopter" };
	// std::array<std::reference_wrapper<std::string>, (std::size_t)1> str {str};

	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>
		generateCalibrationPoints(
			const cv::Mat& unwrapX,
			const cv::Mat& unwrapY,
			float pixelsPer2pi = runtime_flags.disp.wavelength,
			int gridX = 10, //runtime_flags.disp.width
			int gridY = 10, //runtime_flags.disp.height,
			float screenWidth_mm = runtime_flags.disp.width_mm,
			float screenHeight_mm = runtime_flags.disp.height_mm,
			float pixel_pitch_mm = runtime_flags.disp.pixelptich_mm
			);

	std::array<std::string, (std::size_t)8> m_save_keys{ {
		{"wrappedPhasehorizontal"}, {"wrappedPhasevertical"}, {"contrasthorizontal"}, {"contrastvertical"},
		{"baseIntensityhorizontal"}, {"baseIntensityvertical"}, {"unwrappedPhasehorizontal"},
		{"unwrappedPhasevertical"}
	} };

	minmaxloc get_minmaxloc(cv::Mat& mat) const; 

	calibrationData m_calib_data{};

	void shiftStartPhasetoZero(const cv::Mat&, cv::Mat&, float shift);
};


#endif