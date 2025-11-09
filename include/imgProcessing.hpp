#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include "flagHandler.hpp"
#include "imageHandler.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <tuple>
#include "cvDepthTraits.hpp"
#include <filesystem>
#include <ostream>
#include <type_traits>
#include <exception>

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

	// Taken from the c++ library
	// The value given in the constructor allows to differntiat which kind of value (double, float)
	// should be used in the later processing. 
	struct T
	{
		// Be carefull, here non type parater used, because of function template equivalence if not. 
		// std::enable_if does hold has in first parameter condition, second paramter is value thy is given back
		// if condition is true. so ... typename <std::enable_if<condition, type>> = true, so the tyename is assgined the type
		// and directly set to true as a non type parameter. This allows for a conditional non type parameter assignement which 
		// is not marked as the same between the first and second constructor. 
		enum class Type { error_t, float_t, double_t };
		std::array<std::string, std::size_t(3)> type_string{"Error", "float", "double"};
		Type type;
		template<typename Floating,
			typename std::enable_if<std::is_floating_point<Floating>::value, bool>::type = true>
		T(Floating) : type((std::is_same<Floating, double>::value) ? (Type::double_t) : (Type::float_t)) {} // error: treated as redefinition

		template<typename Floating,
			typename std::enable_if<!std::is_floating_point<Floating>::value, bool>::type = true>
		T(Floating) : type(Type::error_t) {}

		constexpr bool is_float()  const { return type == Type::float_t; }
		constexpr bool is_double() const { return type == Type::double_t; }
		constexpr bool is_valid()  const { return type != Type::error_t; }
	};

	

	template<typename Type>
	ImageProcessing(Type val) {
		++instance_counter;
		try {
			if (instance_counter > 1)
				throw std::exception("Only one instance of ImageProcessing allowed");

			T type_info(Type{});
			if (type_info.is_double()) {
				std::cout << "double Path \n";
				setup_matrice<double>();
			}

			else if (type_info.is_float()) {
				std::cout << "Float Path \n";
				setup_matrice<float>();
			}
			// Check runtime type tag
			if (!type_info.is_valid()) {
				throw std::exception("Not a valid type. ImageProcessing must be a floating Point!");
			}
			// Optionally: store it as a member for later use
			current_type = type_info.type;
		}
		catch (std::exception& e) { std::cout << "EXCEPTION " << e.what() << std::endl;
		throw std::runtime_error(e.what());
		}
	}
	// 0 = error , 1 = float, 2 = double 
	T::Type current_type{ T::Type::error_t };
	

	void gray_value_calib(std::vector<cv::Mat>&);

	void wrapped_phase();

	cv::Mat createMask();

	void bayerToGray();

	void unwrapped_phase();

	void goldsteinUnwrap();

	// This function saves uncompressed images to a .xml file. This allows for storing of floating point values. 
	void saveImages(const std::string& path);
	
	// Saves all images from the acquisation and unwrap in png format by first converting and normalizing
	void saveImages_png(const std::string& path);

	void load_frames(const std::string& path);

	void load_calib(std::string path);

	void calc_reproject_error(bool visualizing);

	std::array<cv::Mat, (std::size_t) 2> m_baseIntensity;
	std::array<cv::Mat, (std::size_t) 2> m_contrast;
	std::array<cv::Mat, (std::size_t) 2> m_phase;

	std::vector<cv::Scalar_<double>>& get_mean_values() {
		return m_mean_grayValues;
	}

	// void convertToFloat(std::vector);
	~ImageProcessing();
	ImageProcessing(const ImageProcessing&) = delete;
	ImageProcessing& operator=(const ImageProcessing&) = delete;
	ImageProcessing(ImageProcessing&&) = delete;
	ImageProcessing& operator=(ImageProcessing&&) = delete;

	// Moves AcquisitionFrames from acquisitionworker to the imgProcessing class
	void assginFrames(std::vector<cv::Mat>&& mat) {
		m_frames = mat;
	}
	
	//template<typename func, typename... Mats>
	//static auto forEachPixel(func, Mats&&... mats);

	struct minmaxloc {
		double minval{}, maxval{};
		cv::Point minloc{}, maxloc{};

		friend std::ostream& operator<<(std::ostream& out, minmaxloc loc) {
			std::cout << "Minimum value : " << loc.minval << " at location: " << loc.minloc << '\n' <<
				"Maximum value : " << loc.maxval << "at location: " << loc.maxloc << '\n';
			return out;
		}
	};
	//The .png frames are stored here.
	[[maybe_unused]] std::vector<cv::Mat> m_reprojection_error_img;
private:	
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
	
	template<typename T>
	typename std::enable_if<std::is_floating_point<T>::value, void>::type
		setup_matrice();
	

	//template<int Depth, typename Func, typename... Mats>
	//static cv::Mat forEachPixelImpl(const Func& func, Mats&&... mats);

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

	// Here the floating point values are stored. 
	std::array < cv::Mat, static_cast<std::size_t>(2)> m_reprojection_error{};

	// A type-dependent name is one whose existence or meaning as a type can only be known after template substitution.
	// This is important here. ís_vec_or_array formulates a condition in it´s static variable ::value -> a constant expression
	// Additional typename is NEVER necessary fo expressions, since expression never return types just values. 
	// Expression: any peace of code that can be evaluated to a value.
	// A additional "typename" is only necessary is if the thing we are trying to get is a type and it is not clear - depending on the type -
	// if the type does exist. 
	// The template enabel if:
	// 
	// template <bool B, typename T = void>
	// struct enable_if {}; // primary template — EMPTY
	// 
	// These have partial template specializations:
	// template <typename T>
	// struct enable_if<true, T> { using type = T; };
	// 
	// Therefore: 
	// std::enable_if<true,T>::type -> is not typedependent just because the condition is specialized to true, therefor the value ::type does exist
	// 
	// std::eanble<cond,T>::type -> a typename is necessary because the existenz of the type is chained to value of cond
	// 
	//First time using real typtraits 
	template <typename Container>
	typename std::enable_if<is_vec_or_array<Container>::value, void>::type
		save_png(const std::filesystem::path& path, const Container& c);
		
	template <typename Tpoints, typename Tmat>
	typename std::enable_if<
		std::is_floating_point<Tpoints>::value&&
		std::is_floating_point<Tmat>::value>::type
		project2to3d(
			double* sumSq,
			double* maxErr,
			const std::pair<std::vector<cv::Point_<Tpoints>>, std::vector<cv::Point3_<Tpoints>>>& calib_points,
			std::vector<cv::Vec<Tpoints, 3>>& projected,
			const cv::Mat_<Tmat>& cam_matrix,
			const cv::Mat_<Tmat>& dist_Coeffs,
			const cv::Mat_<Tmat>& rotation_in,
			const cv::Mat_<Tmat>& translation_in);
};


// CV_32F = 5
// CV_64F = 6
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
ImageProcessing::setup_matrice() {
	int type_check{ (std::is_same<T,double>::value) ? CV_64F : CV_32F };
	for (auto& m : m_s1) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_s2) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_s3) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_baseIntensity) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_contrast) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_phase) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_wrapped_phase) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_unwrapped_phase) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
	for (auto& m : m_reprojection_error) m = cv::Mat::zeros(runtime_flags.camera_data.pixel_y, runtime_flags.camera_data.pixel_x, type_check);
}


// one needs a comile time check for that
// is_vec_or_array is stored in cvDeptTraits.hpp
template <typename Container>
typename std::enable_if<is_vec_or_array<Container>::value, void>::type
ImageProcessing::save_png(const std::filesystem::path& path, const Container& c) {
	for (std::size_t i{}; i < c.size(); ++i) {
		std::string path_final = (path / (std::to_string(i) + ".png")).string();
		cv::imwrite(path_final, normalize(c[i]));
	}
}



#endif