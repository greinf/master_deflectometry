#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <tuple>
#include "cvDepthTraits.hpp"
#include <filesystem>
#include <ostream>
#include <type_traits>
#include <exception>
#include <variant>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/traits.hpp>
#include "GrayCalibVector.hpp"
#include "utils.hpp" 
#include "RowPolicy.hpp"
#include "enums.hpp"

using CalibPairF = std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>;
using CalibPairD = std::pair<std::vector<cv::Point2d>, std::vector<cv::Point3d>>;
class ImageStore;

class ImageProcessing {
public:

	// -----------------------------  Depracted!!! ------------------------------------------
	
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
	ImageProcessing(Type val, std::shared_ptr<ImageStore> imgStore)
		:m_imgStore(imgStore)
	{
		try {
			
			T type_info(Type{});
			if (type_info.is_double()) {
				std::cout << "double Path \n";
			}

			else if (type_info.is_float()) {
				std::cout << "Float Path \n";
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
	
	// ---------------------------------------   Depracted End ----------------------------------------------------------

	ImageProcessing(ImageStore& img_store) :
		m_imgStore{ img_store } {
	}

	std::vector<std::pair<double,double>> find_ParallelogramCorners(const cv::Mat& bin);

	std::vector<cv::Point3d> prepareDisplayPoints(
		const std::vector<cv::Point3d>& displayLocalPoints,
		const cv::Mat& housholder,
		const cv::Mat& tvec_mirror);

	std::vector<cv::Mat> manual_phaseUnwrapRef(
		const std::vector<cv::Mat>& wrapped,
		const cv::Mat& mask,
		const double wavelength);
	
	void backToWorld(
		cv::Mat& rvec_w,
		cv::Mat& tvec_w,
		const cv::Mat& rvec_c,
		const cv::Mat& tvec_c,
		const cv::Mat& housholder,
		const cv::Mat& cam_mirror_rvec,
		const cv::Mat& cam_mirror_trans);

	void calculatehousholder(const cv::Mat& rvec_mirror, const cv::Mat& tvec_mirror,
		cv::Mat& tvec_virt, cv::Mat& H);

	std::vector<cv::Point3d> transformPointsToMirrorWorld(
		const std::vector<cv::Point3d>& realPoints,
		const cv::Mat& housholder,
		const cv::Mat& tvec_mirror,
		const cv::Mat& rvec_cam_mir);

	std::vector<cv::Mat> remapCameraToScreen(
		const std::vector<cv::Mat>& image,
		const std::vector<cv::Mat>& mapping_img
	);

	cv::Mat createHomographyFromGrayCode(
		const std::vector<cv::Mat>&,
		const cv::Size& sz
	);

	std::vector<cv::Mat> createMappingfromHomography(
		const cv::Mat& homography,
		const cv::Size& sz);

	cv::Mat remapCameraToScreen(
		const cv::Mat& img,
		const std::vector<cv::Mat>& mapping_img
	);
	

	// Input
	// 1. image of type == CV_64F, 2. double max allowed value in image 3. min allwed value in image
	// Output cv::Mat of type cv_64F and size of img
	// The Method aborts if the Values in the image are higher than max val or lower than minval !
	cv::Mat quantizeImage(
		const cv::Mat& img,
		const double max_allowed = 255.0,
		const double min_allowed = 0.0
	);


	template<typename RowPolicy = UniformRowsCols>
	std::vector<cv::Mat> do_gamma_distortion(
		const double gamma,
		const std::vector<cv::Mat>& images,
		RowPolicy = {})
	{
		std::vector<cv::Mat> distorted;
		for (const auto& img : images) {
			distorted.push_back(do_gamma_distortion(gamma, img, RowPolicy{}));
		}
		return distorted;
	}


	// Input 1: double value for gamma distortion must be >=0
	// Input 2: Image on which to perfrom the operation. 
	// If img.type() != CV64F it is converted! 
	// Input 3: Row Policy: if UnifromRowsCols{} 
	// operation will be perfromed for one row and repeated over the image
	// Ouptput: Picture with gamma distortion of type CV_64F
	template<typename RowPolicy = UniformRowsCols>
	cv::Mat do_gamma_distortion(
		const double gamma,
		const cv::Mat& img,
		RowPolicy = {})
	{
		static_assert(is_row_policy_v<RowPolicy>,
			"RowPolicy must be PerElement or UniformRowsCols");
		CV_Assert(gamma >= 0);
		CV_Assert(!img.empty());
		CV_Assert(img.channels() == 1);
		cv::Mat img64, imgGamma;

		if (img.type() != CV_64F)
			img.convertTo(img64, CV_64F);
		else img64 = img;
		if (gamma == 1) return img64;

		if constexpr (std::is_same<RowPolicy, UniformRowsCols>::value) {
			bool unifrom_cols{ true };
			for (int row = 0; row < img64.rows; ++row) {
				const double* start = img64.ptr<double>(row);
				const double* middle = img64.ptr<double>(row) + (img64.cols / 2);
				const double* end = img64.ptr<double>(row) + (img64.cols - 1);
				if (*start != *middle || *middle != *end) {
					unifrom_cols = false;
					break;
				}
			}
			switch (unifrom_cols) {
				case(true): {
					cv::Mat col(img64.rows, 1, CV_64F);
					for (int row = 0; row < img64.rows; ++row) {
						*col.ptr<double>(row) = 
							255.0 * std::pow(*img64.ptr<double>(row)/255.0, gamma);
					}
					imgGamma = cv::repeat(col, 1, img64.cols);
					break;
				}
				case(false): {
					cv::Mat row(1, img64.cols, CV_64F);
					double* row_ptr = row.ptr<double>(0);
					double* img_ptr = img64.ptr<double>(img64.rows / 2);
					for (int col = 0; col < img64.cols; ++col) {
						row_ptr[col] = 255.0 * std::pow(img_ptr[col]/ 255.0, gamma);
					}
					imgGamma = cv::repeat(row, img64.rows, 1);
					break;
				}
			}
		}
		else {
			imgGamma.create(img64.size(), CV_64F);
			for (int row = 0; row < img64.rows; ++row) {
				for (int col = 0; col < img64.cols; ++col) {
					imgGamma.ptr<double>(row)[col] = 
						255.0 * std::pow(img64.ptr<double>(row)[col]/255.0, gamma);
				}
			}
		}
		
		return imgGamma;
	}

	std::vector<std::pair<double, double>> gray_value_calib(
		const std::vector<cv::Mat>&, 
		int pics_per_val, 
		int stepwidth);

	// Calculates the Centers of the Circles located in a image
	// 1. In - Image with symmetrical Circle grid with type CV_8U
	// 2. In - Masking image of Type CV_8U. Must be same sice as 1. 
	// 3. In - Size of the circle Grid to be expected (Here 20 x 20 circel Grid)
	// Return vector of cv::Vec2f with center coordinates 
	std::vector<cv::Vec2d> getCircleCoordinates(
		const cv::Mat& img,
		const cv::Mat& mask,
		const cv::Size sz
	) const;

	// Creates a vector of Object Coordinates R^3 for Pose solving
	// 1. In - Size of the pattern calibration Pattern
	// 2. In - The distance between to calibration Points on the pattern. 
	// 3. Return a std::vector of pattern_size.area() 
	std::vector<cv::Vec3d> createCalibPatternObjectPoints(
		const cv::Size& pattern_size,
		const double distance
	);

	// Calculates the Normal for the easy case the the ObjectPoints lie
	// on a cartesian grid.
	// Take two orthogonal vector in display coordantes and calcualte the normal
	cv::Vec3d getNormalofPlane(
		cv::Mat_<cv::Vec3d> objectPoints
		);


	cv::Mat_<cv::Vec3d> getReflectedRays(
		const cv::Mat_<cv::Vec3d>& rays,
		const cv::Vec3d& normalVec,
		const cv::Mat& mask
	);


	// Return a std::vector<cv:Mat> with 6 entries
	// 0-1 Wrapped phase (horizontal - vertical)
	// 1-2 Contrast (horizontal - vertical)
	// 3-4 Base Intensity (horizontal - vertical)
	std::vector<cv::Mat> do_wrapped_Phase(
		const std::vector<cv::Mat>&, 
		int n_pics_per_Phase,
		int n_shifts);

	cv::Mat getHomographyMat(
		const std::vector<std::pair<int, int>>& src,
		const std::vector<std::pair<double, double>>& dst,
		const int method = 1
	);

	cv::Mat getHomographyMat(
		const std::vector<cv::Vec2d>&,
		const std::vector<cv::Vec2d>&,
		const int method = 1
	);

	// Create circular mask with given diameter
	// Return mat is row = cols = diameter; type = CV_64F
	cv::Mat createCirculeBinaryMask(
		const int diameter);

	cv::Mat getHomographyMat(
		const RoiBorders<int>,
		const RoiBorders<double>,
		const int method = 1
	);

	void getResponseCurve_perSection(
		const std::vector<cv::Mat>& gray_images,
		GrayCalibVector& borders,
		const int n_pics_perVal,
		const int n_steps
	);

	// Input
	// 1 Coordiantes as Vec2d in (x,y)
	// cam Matrix size (3,3) in floating
	// dist coeffs in floating
	cv::Vec2d newtonSolverdistort(
		const cv::Vec2d& coordiantes,
		const cv::Mat& cam_Matrix,
		const cv::Mat& dist_coeffs
	);

	// Input: 2 Images of the gray calib -> brightes & darkest
	// gray[0] -> darkest; gray[1] brightest
	// Output cv::Mat with size gray[0].size(), gray[0].type() = CV_8U 
	cv::Mat grayCalibMask(
		const std::vector<cv::Mat>& gray);
	
	cv::Vec2d ImageProcessing::undistortImagePts(const cv::Vec2d pts,
		const cv::Mat& K,
		const cv::Mat& distCoeffs);

	std::vector<cv::Vec2d> distortImagePoints(
		const std::vector<cv::Vec2d>&,
		const cv::Mat&,
		const cv::Mat&
	);

	// Functions Fits Surface over the image
	// Input: 1 Image of Type CV_64F, channels() == 1; Image to fit
	// 2 Image of Type CV_8U, channels() == 1, Image for Masking
	// returns: Image with fitted surface at not masked area. 
	cv::Mat fitSurface(
		const cv::Mat& img,
		const cv::Mat& mask 
	);

	// Input takes img.channels() == 2 of CV_64F
	// Function Calculates:
	// error_x = img.at()[0] - col
	// error_y = img.at()[1] - row
	// return cv::Mat channels() == 2 of typ CV_64F
	cv::Mat calcDistortionError(const cv::Mat& img);

	// Calculates Mask from both the contrast pictures
	// Input 
	// vec: 2 Images CV_64F double Horizontal / Vertical
	// treshold: Threshold for binary mask. Encodes at which percentag of the maximum the boundarie is (0 ... 1)
	// Output: CV_8U Picture, same size as Input
	cv::Mat createMask(
		const std::vector<cv::Mat>& vec,
		double threshold,
		bool dilate = false);

	cv::Mat ImageProcessing::undistortImage(const cv::Mat& img,
		const cv::Mat& K,
		const cv::Mat& distCoeffs);

	std::vector<cv::Mat> bayerToGray(
		const std::vector<cv::Mat>&
	);

	std::vector<cv::Mat> unwrapped_phase(
		const std::vector<cv::Mat>& wrapped_phase,
		const cv::Mat& mask);

	void evaluateSection_gray(
		const std::vector<cv::Mat>& images,
		Gray_section_data& data,
		const cv::Mat& homography,
		const cv::Mat& mask
	);

	std::pair<double, double> ImageProcessing::fitLine1D(const std::vector<double>& y);

	// Function takes:
	// wrapped = std::vector<cv::Mat> wrapped Phase horizontal + vertical -> 2 Images type: double
	// contrast = std::vector<cv::Mat> contrast horizontal + vertical -> 2 Images type: double 
	// n_shifts = int Information how many Shifts per 
	std::vector<cv::Mat> manual_phaseUnwrap(
		const std::vector<cv::Mat>& wrapped,
		const cv::Mat& mask		
		);

	std::vector<double> ImageProcessing::extract_Column(const cv::Mat& picture, const cv::Mat& mask, int col = 0);

	std::vector<double> ImageProcessing::extract_Line(const cv::Mat& picture, const cv::Mat& mask, int row = 0);

	std::array<cv::Mat, (std::size_t)2> m_unwrapped_phase{};
	std::array<cv::Mat, (std::size_t)2> m_wrapped_phase{};

	//// Here the floating point values are stored. 
	//std::array < cv::Mat, static_cast<std::size_t>(2)> m_reprojection_error{};

	// void convertToFloat(std::vector);
	~ImageProcessing();
	ImageProcessing(const ImageProcessing&) = delete;
	ImageProcessing& operator=(const ImageProcessing&) = delete;
	ImageProcessing(ImageProcessing&&) = delete;
	ImageProcessing& operator=(ImageProcessing&&) = delete;


	struct minmaxloc {
		double minval{}, maxval{};
		cv::Point minloc{}, maxloc{};

		friend std::ostream& operator<<(std::ostream& out, minmaxloc loc) {
			std::cout << "Minimum value : " << loc.minval << " at location: " << loc.minloc << '\n' <<
				"Maximum value : " << loc.maxval << "at location: " << loc.maxloc << '\n';
			return out;
		}
	};

	std::vector<cv::Vec2d> getRefinedCheckerboardCorner(
		const std::vector<cv::Mat> img,
		const cv::Size sz
	);

	std::vector<cv::Vec2d> harrisCornerDetection(
		const cv::Mat& img,
		const cv::Mat& mask,
		const cv::Size& window_sz,
		const double k1 = 0.04, 
		bool blur = true, 
		bool gaussian_window = true
	);

	std::vector<cv::Vec2d> doTemplateMatching(
		std::vector<cv::Vec2d>& corners,
		const cv::Mat& img,
		const cv::Mat& templ
	);

	std::vector<cv::Vec2d> findExtrema(
		const std::vector<cv::Vec2d>&,
		const std::vector<cv::Mat>&,
		const cv::Size&
	);

	std::vector<cv::Vec2d> findExtrema(
		const std::vector<cv::Vec2d>&,
		const cv::Mat&,
		const cv::Size&
	);

	double bilinearInterpolation(
		const cv::Mat& img,
		const cv::Vec2d& coordintate
	);

	std::vector<cv::Vec2d> doTemplateMatching(
		std::vector<cv::Vec2d>& corners,
		const std::vector<cv::Mat>& img,
		const cv::Mat& templ
	);

	cv::Mat ImageProcessing::getTemplateChess(
		const cv::Mat img,
		const cv::Size sz = { 50,50 }
	);

	// Just a helper function that means over vector before doing the Corner Detection
	// Return std::vector<cv::vec2d> in (y,x)
	std::vector<cv::Vec2d> harrisCornerDetection(
		const std::vector<cv::Mat>& img,
		const cv::Mat& mask,
		const cv::Size& window_sz,
		const double k1 = 0.04, 
		bool blur = true, 
		bool gaussian_window = true
	);

	// Build the openCv method for refining the corners again.
	// This methods allows double values which are prohobited in the openCV method
	std::vector<cv::Vec2d> refineCorner(
		const std::vector<cv::Mat>& img,
		const std::vector<cv::Vec2d>& corners,
		const cv::Size& sz,
		const double eps
	);

	std::vector<cv::Vec2d> refineCorner(
		const cv::Mat& img,
		const std::vector<cv::Vec2d>& corners,
		const cv::Size& sz,
		const double eps
	);

	cv::Vec2d projectPoint_homography(
		const cv::Vec2d& img,
		const cv::Mat& homography
	);

	// 1 Row number
	// 2 Image of type CV_64F, 
	// 3 Image (mask) of type CV_8U
	std::vector<double> extract_Column(
		int x,
		const cv::Mat& img,
		const cv::Mat& mask);


	// 1 Row number
	// 2 Image of type CV_64F, 
	// 3 Image (mask) of type CV_8U
	std::vector<double> extract_Row(
		int x,
		const cv::Mat& img,
		const cv::Mat& mask);

	// Setup for calculating weights for the gain of single color channels -> BayerRG setup
	// Return {B,G,R}
	// Because the Bild was rotated in this context RGB !!!
	std::array<double, (std::size_t)3> doWhiteBalance(
		const std::vector<cv::Mat>&,
		int roi_x,
		int roi_y,
		int roi_width,
		int roi_height);

	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> do_calibration_Points(
		const std::vector<cv::Mat>& unwrapped,
		const cv::Mat& mask,
		const cv::Vec2d& refPoint,
		const double wavelength,
		const int gridX,
		const int gridY,
		const double pixel_pitch_mm
	);

	cv::Mat do_reprojection_error(
		const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>>& caliPoints,
		const cv::Mat& caliMatrix,
		const cv::Mat& distCoeffs,
		const cv::Mat& rvec,
		const cv::Mat& tvec,
		double* sqrtErr,
		double* maxErr,
		const cv::Mat& mask
		);

	cv::Vec2d distortImagePoints(
		const cv::Vec2d& imgPts,
		const cv::Mat& calimatrix,
		const cv::Mat& distcoeffs);


	// Input
	// 1 3x3 Camera Matrix of floating point CV_32F || CV_64F
	// 2 dist coeffs matrix 1x5 || 1x6 of CV_32F || cv_64F
	// 3 cv::Mat of pixel in sensor coordiantes pixels as cv::Vec2d 
	// Output cv::Mat_<cv::Vec3d> of size pixel_y x pixel_x
	cv::Mat calulateRays(
		const cv::Mat& camera,
		const cv::Mat& dist_coeffs,
		const cv::Mat& sensor_coordinates
	);

	// Calculates a coordinated image in Object coordiantes in the display coordiante system
	// The origin is placed in the center and can be moved with shift_x and y
	// pixelPitch is the scaling factor for the coordinates 
	// functoin return a cv::Mat_<cv::Vec3d> 
	// DO NOT USE THE SHIFT_X SHIFT_Y ALWAYS 0
	cv::Mat calcCoordinateImage(
		const cv::Size& sz,
		const double pixel_pitch_disp,
		const double shift_x,
		const double shift_y
	);

	// Method calculates a rotation of a coordianted grid stored as cv::Mat_<cv::Vec3d>
	// 1. [In] - Image of type cv::Vec3d
	// 2. [In] - cv::Vec3d 
	// 3. [In] - AngleInterpretation: 
	// if euler -> euler xy cv::Vec3d(rotation around x, rotatoin around y, rotaton around z is discarded)
	// if Rodrigeus -> the Vector is interpreted as Rodrigeuz vector. See openCV
	// 4. Return the a cv::Mat_<cv::Vec3d> with for each entry rotated coordiantes. 
	cv::Mat rotateCoordinatedGrid(
		const cv::Mat& img,
		const cv::Vec3d& rotVector,
		Rotation rot = Rotation::eulerxy
	);

	// Takes 1 grid (cv::Mat_<Vec3d> of coordinate points. 
	// 2. cv::vec3d the shift Vector to shift the Coordiate Grid
	// Return a cv::Mat_<cv::Vec3d> with all vector shifted about the shift vector
	cv::Mat shiftCoordinateGrid(
		const cv::Mat_<cv::Vec3d>& img,
		const cv::Vec3d& shift
	);

	// Input
	// 1 cv::Mat_<cv::Vec3d> of rays
	// 2 cv::Mat_<cv::vec3d> of Objekt Points
	// return a cv::Mat<Vec3d> of the all impact points.
	// No impact is (0,0,0)
	// Rays can be masked as invalid rays when set to {-1,-1,-1}
	std::vector<cv::Mat> calculateHitPoints(
		const cv::Mat_<cv::Vec3d> rays,
		const cv::Mat_<cv::Vec3d> display_coordiantes
	);

	cv::Mat mapHitPointsToDisplayCoords(
		const cv::Mat& hitpoints,        // CV_64FC3, size = rays.size()
		const cv::Mat& display_points    // CV_64FC3, size = (H,W) of display raster
	);

	cv::Mat mean(const std::vector<cv::Mat>&);

private:	
	//New class to hold the data
	ImageStore& m_imgStore;

	// Creates a coordiante Image in the coordiante System of the image in 3d
	// Input 
	// 1 Size of the Image
	// 2 PixelPitch of the physikal display
	// 3 If shit_x = 0 origina is in center of the image x > 0 shifts right
	// 4 shift_y > 0 shifts down
	// 5 angle_x image is tilted in X. Image coordinates system stands stil
	// 6 angle_x image is tilted in Y. IMage coordinate system stand stil
	// Returns cv::Mat with 4 channels (x,y,z,1) of CV_64F
	cv::Mat angle_camera_toScreenNormal(
		const cv::Size& sz,
		const double pixel_pitch_disp,
		const double shift_x,
		const double shift_y,
		const double angle_x,
		const double angle_y,
		const double distance
	);

	
	
	
	std::vector<std::pair<double, double>> orderTLTRBRBL_sumdiff(const std::vector<std::pair<double, double>>& p);

	void checkHomogrpahy(
		const cv::Mat& img,
		const cv::Mat& homography,
		const cv::Vec2d& coord
	);

	cv::Mat m_mask;
	
	void unwrap_row(
		const cv::Mat& wrapped,
		cv::Mat& unwrapped,
		const cv::Mat& mask,
		int row);

	void unwrap_row(
		const cv::Mat& wrapped,
		const cv::Mat& wrapped_reference,
		cv::Mat& unwrapped,
		const double wavelength,
		const cv::Mat& mask,
		int row);

	void unwrap_column(
		const cv::Mat& wrapped,
		cv::Mat& unwrapped,
		const cv::Mat& mask,
		int colunn);

	void unwrap_column(
		const cv::Mat& wrapped,
		const cv::Mat& wrapped_reference,
		cv::Mat& unwrapped,
		const double wavelength,
		const cv::Mat& mask,
		int column);

	std::string path{ "C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out" };

	std::array<std::string, 7> m_reprojection_error_string{ "SqrtError_x", "SqrtError_y", "Median_x", "Median_y",
		"Max_Error_x", "Max_Error_y", "Error_Vectors"};
	
	cv::Mat applyMask(const cv::Mat&, const cv::Mat&, float shift = 0.0f);

	minmaxloc get_minmaxloc(const cv::Mat& mat) const; 

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
		

};


// one needs a comile time check for that
// is_vec_or_array is stored in cvDeptTraits.hpp
template <typename Container>
typename std::enable_if<is_vec_or_array<Container>::value, void>::type
ImageProcessing::save_png(const std::filesystem::path& path, const Container& c) {
	for (std::size_t i{}; i < c.size(); ++i) {
		std::string path_final = (path / (std::to_string(i) + ".png")).string();
		
		if (c[i].channels() == 3) {
			cv::imwrite(path_final, c[i]);
			continue;
		}

		cv::imwrite(path_final, normalize(c[i]));
	}
}

#endif


// Depracted code 


/*
//Back inserter example code
template <class Container>
class back_insert_iterator {
protected:
	Container* container; // pointer to your vector
public:
	explicit back_insert_iterator(Container& c) : container(std::addressof(c)) {}

	back_insert_iterator& operator=(const typename Container::value_type& value) {
		container->push_back(value);
		return *this;
	}

	back_insert_iterator& operator*() { return *this; }
	back_insert_iterator& operator++() { return *this; }
	back_insert_iterator operator++(int) { return *this; }
};

// Saves all images from the acquisation and unwrap in png format by first converting and normalizing
	void saveImages_png(const std::string& path);



Examamle usage of:
std::variant -> runtime type variability and 
tempalte deduction guides -> compile time variability. 
#include <iostream>
#include <vector>
#include <variant>
#include <type_traits>

// -----------------------------------------------------------
// 1) Clean compile-time design with CTAD (deduction guides)
// -----------------------------------------------------------
template<typename T>
struct CalibPair {
	std::vector<T> image;
	std::vector<T> object;
};

// Deduction guides — connect constructor arguments to type:
// Just these deduction guides allow the usage of auto as return type !!!
CalibPair(std::vector<float>,  std::vector<float>)  -> CalibPair<float>;
CalibPair(std::vector<double>, std::vector<double>) -> CalibPair<double>;

// A templated factory that just uses the same T throughout:
template<typename T>
auto makeCalibPair() {
	std::vector<T> a{1,2,3};
	std::vector<T> b{4,5,6};
	return CalibPair{a, b}; // CTAD picks CalibPair<float> or CalibPair<double>
}

// -----------------------------------------------------------
// 2) Runtime style using std::variant
// -----------------------------------------------------------
using CalibPairF = std::pair<std::vector<float>,  std::vector<float>>;
using CalibPairD = std::pair<std::vector<double>, std::vector<double>>;
using CalibVariant = std::variant<CalibPairF, CalibPairD>;

template<typename T>
CalibVariant makeCalibVariant() {
	std::vector<T> a{1,2,3};
	std::vector<T> b{4,5,6};
	if constexpr (std::is_same_v<T, float>)
		return CalibPairF{a, b};
	else
		return CalibPairD{a, b};
}

// -----------------------------------------------------------
// main()
// -----------------------------------------------------------
int main() {
	//   Compile-time style
	auto cp_f = makeCalibPair<float>();
	auto cp_d = makeCalibPair<double>();
	std::cout << "CTAD type float:  "  << typeid(cp_f).name() << "\n";
	std::cout << "CTAD type double: "  << typeid(cp_d).name() << "\n";

	//  Runtime-style variant
	auto cv_f = makeCalibVariant<float>();
	auto cv_d = makeCalibVariant<double>();
	std::cout << "Variant holds float?  "
			  << std::holds_alternative<CalibPairF>(cv_f) << "\n";
	std::cout << "Variant holds double? "
			  << std::holds_alternative<CalibPairD>(cv_d) << "\n";
}

//	void goldsteinUnwrap();


// Old code where tried to template this hole class - bad idea 

// Templated function for generating the calibration points
	// Also the return type is templated for my variablity
	template<typename T>
	auto generateCalibrationPoints(
		const cv::Mat& unwrapX,
		const cv::Mat& unwrapY,
		float pixelsPer2pi = runtime_flags.disp.wavelength,
		int gridX = 10, //runtime_flags.disp.width
		int gridY = 10, //runtime_flags.disp.height,
		float screenWidth_mm = runtime_flags.disp.width_mm,
		float screenHeight_mm = runtime_flags.disp.height_mm,
		float pixel_pitch_mm = runtime_flags.disp.pixelptich_mm
	)
	{
		CV_Assert(unwrapX.size() == unwrapY.size());
		CV_Assert((unwrapX.type() == CV_32FC1 && unwrapY.type() == CV_32FC1) ||
			(unwrapX.type() == CV_64FC1 && unwrapY.type() == CV_64FC1));
		int screenPxWidth{ runtime_flags.disp.width };
		int screenPxHeight{ runtime_flags.disp.height };
		std::vector<cv::Point_<T>> imagePoints;
		std::vector<cv::Point3_<T>> objectPoints;


		// Define spacing across image (camera pixels)
		T stepX = static_cast<T>(unwrapX.cols) / (gridX);  //+1
		T stepY = static_cast<T>(unwrapX.rows) / (gridY);  //+1

		//Debug Copy
		cv::Mat debug = m_mask.clone();
		cv::cvtColor(debug, debug, cv::COLOR_GRAY2BGR);

		for (int gy = 0; gy < gridY; ++gy) {
			//if (stepY == 1) --gy;
			int dy = static_cast<int>(gy * stepY);
			if (dy < 0 || dy >= unwrapX.rows)
				continue;

			const T* row_ptr_X = unwrapX.ptr<T>(dy);
			const T* row_ptr_Y = unwrapY.ptr<T>(dy);
			for (int gx = 0; gx < gridX; ++gx) {
				//if (stepX == 1) --gx;
				int px = static_cast<int>(gx * stepX);
				//int py = static_cast<int>(gx * stepY);

				if ((*(row_ptr_X + px) == -10.0f) || (*(row_ptr_Y + px) == -10.0f)) {
					//std::cout << "Jump over \n";
					continue; // Ye haa -> pointer arithmetic bitches
				}

				if (px < 0 || dy < 0 || px >= unwrapX.cols || dy >= unwrapX.rows) {
					std::cout << "Something must have went terribly wrong: GenerateCalibrationPoints() \n";
					continue;
				}

				//Can be used to acces array element, but documentatoins says it is slow.
				//float phiX = unwrapX.at<float>(py, px);
				//float phiY = unwrapY.at<float>(py, px);
				// Get the phase value in each pictures for the same pixel if valid.
				T phiX = (*(row_ptr_X + px));
				T phiY = (*(row_ptr_Y + px));

				// Maximum possible phase values
				T max_phase_value_u = (static_cast<T>(runtime_flags.disp.width) / pixelsPer2pi) * CV_2PI;
				T max_phase_value_v = (static_cast<T>(runtime_flags.disp.height) / pixelsPer2pi) * CV_2PI;

				//Check if phase value is in logical range. These boarder need to be less strict
				constexpr T phaseMargin = static_cast<T>(2.0 * CV_PI); // one full period tolerance
				if (phiX > max_phase_value_u + phaseMargin || phiY > max_phase_value_v + phaseMargin) {
					std::cerr << "Phase out of range at (" << px << ", " << dy << "): "
						<< phiX << " / " << phiY << "\n";
					continue;
				}

				// Convert unwrapped phase  screen coordinates in subpixel (float, float) values.
				// Be carefull -> the phase direction is: <- (left for horizontal) and î (upwards for vertical)
				// The coordinate system should be in top left so we have to convert back.
				T u_s = (phiX / (2.0f * CV_PI)) * pixelsPer2pi;
				T v_s = (phiY / (2.0f * CV_PI)) * pixelsPer2pi;

				u_s = screenPxWidth - u_s;
				v_s = screenPxHeight - v_s;

				// Convert to millimetres using screen dimensions
				T Xs = u_s * pixel_pitch_mm;
				T Ys = v_s * pixel_pitch_mm;

				// Store results
				imagePoints.emplace_back(static_cast<T>(px), static_cast<T>(dy));
				objectPoints.emplace_back(Xs, Ys, T(0));
				cv::drawMarker(debug, cv::Point(px,dy), cv::Scalar(0, 255, 0));
			}
		}

		//cv::imshow("Show marker Points on mask", debug);
		//cv::waitKey(0);

		// So here a constexpr if is used -> This determines a unambigous return type which MUST be determined
		// for each template clearly.
		if constexpr (std::is_same<T, double>::value) return CalibrationPoints<double> {objectPoints, imagePoints};
		else return CalibrationPoints<float>{objectPoints, imagePoints};
	}

	// Al lot of old code that was used for the old templated imgProcessing class do not use. 
		 
template <typename T>
typename std::enable_if<
	std::is_floating_point<T>::value,
	typename std::conditional<
	std::is_same<T, double>::value,
	ReprojectionError<double>,
	ReprojectionError<float>
	>::type
>::type

project2to3d(
	const CalibrationPoints<T> projected,
	cv::Mat& cam_matrix,
	cv::Mat& dist_Coeffs,
	cv::Mat& rotation_in,
	cv::Mat& translation_in)
{

	if constexpr (std::is_same<T, double>::value) {
		rotation_in.convertTo(rotation_in, CV_64F);
		translation_in.convertTo(translation_in, CV_64F);
	}
	else {
		rotation_in.convertTo(rotation_in, CV_32F);
		translation_in.convertTo(translation_in, CV_32F);
	}

	//There is no constructor from cv::Mat -> cv::Vec<T,3> even if it would fit
	// remeber translation_in (3,1) y,x
	cv::Vec<T, 3> translation_i(translation_in.at<T>(0),
		translation_in.at<T>(1),
		translation_in.at<T>(2));

	cv::Mat_<T> rotation_ex;
	cv::Rodrigues(rotation_in, rotation_ex);

	cv::Mat_<T> cam_inv{ cam_matrix.inv() };
	cv::Mat_<T> rot_inv{ rotation_ex.inv() };


	std::vector<cv::Point_<T>> undistorted_img{};
	cv::undistortImagePoints(projected.imagePoints, undistorted_img, cam_matrix, dist_Coeffs);

	//CV_Assert(undistorted_img.size() == projected.imagePoints.size());
	//for (std::size_t i = 0; i < undistorted_img.size(); ++i) {
	//	std::cout << "Image Point: " << projected.imagePoints[i] <<
	//		"  Undistorted Point normalized?: " << undistorted_img[i] << '\n' << 
	//		"Undistorted Point times pix" << undistorted_img[i].x*runtime_flags.camera_data.pixel_x << ',' << 
	//		undistorted_img[i].y*runtime_flags.camera_data.pixel_y << '\n';
	//}

	//Extend the image_pic (x,y) -> (x,y,1)
	std::vector<cv::Point3_<T>> homogen_imagePoints_undis{};
	std::transform(undistorted_img.begin(), undistorted_img.end(),
		// small exmaple for back_inserter below
		std::back_inserter(homogen_imagePoints_undis),
		[](const cv::Point_<T>& img_p) -> cv::Point3_<T>
		{ return cv::Point3_<T>(img_p.x, img_p.y, 1); }
	);

	//Error Calculation

	double sqrt_error_x{};
	double sqrt_error_y{};
	double median_x{}, median_y{};
	double max_error_x{};
	double max_error_y{};

	std::vector<cv::Vec<T, 2>> error_vec{};
	cv::Mat_<T> cam_center_mat = (-rot_inv) * translation_i;
	cv::Vec<T, 3> camera_center(cam_center_mat(0), cam_center_mat(1), cam_center_mat(2));

	cv::Mat_<T> transform_inv = rot_inv * cam_inv;

	cv::Mat error_visualizer2(m_mask.size(),
		CV_MAKETYPE(cv::DataType<T>::depth, 2),
		cv::Scalar(0, 0));

	for (std::size_t count = 0; count < homogen_imagePoints_undis.size(); ++count) {
		cv::Mat_<T> pt = (cv::Mat_<T>(3, 1) << homogen_imagePoints_undis[count].x,
			homogen_imagePoints_undis[count].y,
			homogen_imagePoints_undis[count].z);

		cv::Mat_<T> ray_mat = transform_inv * pt;
		cv::Vec<T, 3> ray_inv(ray_mat(0), ray_mat(1), ray_mat(2));

		T s = -camera_center[2] / ray_inv[2];
		cv::Vec<T, 3> Pw = camera_center + s * ray_inv;
		cv::Vec<T, 3> diff = Pw - cv::Vec<T, 3>(projected.objectPoints[count].x,
			projected.objectPoints[count].y,
			projected.objectPoints[count].z);
		double error = cv::norm(diff);
		// std::cout << error << " [mm] 0,2745mm pixelpitch" << '\n';
		sqrt_error_x += diff[0] * diff[0];
		sqrt_error_y += diff[1] * diff[1];
		// sqrt_error += error * error;
		// if (error > max_error) max_error = error;
		if (std::abs(diff[0]) > max_error_x) max_error_x = std::abs(diff[0]);
		if (std::abs(diff[1]) > max_error_y)  max_error_y = std::abs(diff[1]);
		// A vector just with the errors for median 
		error_vec.emplace_back(cv::Vec<T, 2>(diff[0], diff[1]));
		int ix = cvRound(projected.imagePoints[count].x);
		int iy = cvRound(projected.imagePoints[count].y);

		if (ix >= 0 && ix < error_visualizer2.cols &&
			iy >= 0 && iy < error_visualizer2.rows)
		{
			error_visualizer2.at<cv::Vec<T, 2>>(iy, ix) = cv::Vec<T, 2>(diff[0], diff[1]);
		}
	}

	// MixChannels
	cv::Mat out[2] = {
		cv::Mat(error_visualizer2.size(), CV_MAKETYPE(cv::DataType<T>::depth, 1)),
		cv::Mat(error_visualizer2.size(), CV_MAKETYPE(cv::DataType<T>::depth, 1))
	};
	int from_to[] = { 0,0, 1,1 };
	cv::mixChannels(&error_visualizer2, 1, out, 2, from_to, 2);

	std::vector<T> median_calculation_x{};
	std::vector<T> median_calculation_y{};

	// Vector of all errors in each direction
	for (std::size_t i = 0; i < error_vec.size(); ++i) {
		median_calculation_x.push_back(std::abs(error_vec[i][0]));
		median_calculation_y.push_back(std::abs(error_vec[i][1]));
	}

	std::size_t n{ median_calculation_x.size() };

	//Directly sort the median calculation vector
	std::sort(median_calculation_x.begin(), median_calculation_x.end());
	std::sort(median_calculation_y.begin(), median_calculation_y.end());

	if (n > 0) {
		if (n % 2 == 0) {
			// even
			median_x = 0.5 * (median_calculation_x[n / 2 - 1] + median_calculation_x[n / 2]);
			median_y = 0.5 * (median_calculation_y[n / 2 - 1] + median_calculation_y[n / 2]);
		}
		else {
			// odd
			median_x = median_calculation_x[n / 2];
			median_y = median_calculation_y[n / 2];
		}
	}


	// use xy[0]/xy[1] (or out[0]/out[1]) below:
	m_reprojection_error[0] = out[0].clone();
	m_reprojection_error[1] = out[1].clone();

	

	// *mainly to see the error of the grayvalue calibration (periodic errors)*

	assert(projected.imagePoints.size() == error_vec.size() && "Errorvec size() and image_points must be the same. \n");

	//std::vector<cv::Mat> xy(2);
	//cv::split(error_visualizer2, xy);

	minmaxloc x_direc{ get_minmaxloc(out[0]) };
	minmaxloc y_direc{ get_minmaxloc(out[1]) };
	double x_range = x_direc.maxval - x_direc.minval;
	double y_range = y_direc.maxval - y_direc.minval;
	int x_max, y_max;
	(x_range > y_range) ? (x_max = 255, y_max = static_cast<int>(255 * (y_range / x_range))) :
		(x_max = static_cast<int>(255 * (x_range / y_range)), y_max = 255);
	//In this code both images are just normalized to their given range. Therefore the color map gives no information a
	cv::Mat visual_mask = ((out[0] != 0) | (out[1] != 0));
	cv::Mat xNorm, yNorm;
	// First normalization, than conversion to the needed Datatype !!!!
	cv::normalize(out[0], xNorm, 0, x_max, cv::NORM_MINMAX);
	cv::normalize(out[1], yNorm, 0, y_max, cv::NORM_MINMAX);
	xNorm.convertTo(xNorm, CV_8U);
	yNorm.convertTo(yNorm, CV_8U);
	cv::Mat xcolor, ycolor;
	cv::applyColorMap(xNorm, xcolor, cv::COLORMAP_TURBO);
	cv::applyColorMap(yNorm, ycolor, cv::COLORMAP_TURBO);
	//Set to zero, if the mask is not true. So just extra to set all elements that are masked to zero. 
	xcolor.setTo(cv::Scalar(0, 0, 0), ~visual_mask);
	ycolor.setTo(cv::Scalar(0, 0, 0), ~visual_mask);

	m_reprojection_error_img.push_back(xcolor.clone());
	m_reprojection_error_img.push_back(ycolor.clone());

	

	sqrt_error_x = std::sqrt(sqrt_error_x / homogen_imagePoints_undis.size());
	sqrt_error_y = std::sqrt(sqrt_error_y / homogen_imagePoints_undis.size());;

	return ReprojectionError(sqrt_error_x, sqrt_error_y, median_x, median_y, max_error_x, max_error_y, std::move(error_vec));
}


// Templated struct to hold the Calibration points -> generateCalibrationPoints<>()
	template<typename T>
	struct CalibrationPoints {
		std::vector<cv::Point3_<T>> objectPoints{};
		std::vector<cv::Point_<T>> imagePoints{};

		constexpr CalibrationPoints() = default;

		constexpr CalibrationPoints(
			std::vector<cv::Point3_<T>> obj,
			std::vector<cv::Point_<T>> img)
			: objectPoints(std::move(obj)), imagePoints(std::move(img)) {
		}
	};
	//Deduction guide. This lead to automatic template deduction and a unambigous return type of generateCalibration points
	CalibrationPoints(std::vector<cv::Point_<T>>, std::vector<cv::Point3f>)->CalibrationPoints<float>;
	CalibrationPoints(std::vector <cv::Point_<T>>, std::vector<cv::Point3d>)->CalibrationPoints<double>;


	template<typename T>
	struct ReprojectionError {
		double sqrt_Error_x{};
		double sqrt_Error_y{};
		double median_x{};
		double median_y{};
		double max_Error_x{};
		double max_Error_y{};
		std::vector<cv::Vec<T, 2>> reproject_error_vec{};

		constexpr ReprojectionError() = default;

		//Constructor
		constexpr ReprojectionError(
			double sqrt_Err_x,
			double sqrt_Err_y,
			double media_x,
			double media_y,
			double max_Err_x,
			double max_Err_y,
			std::vector<cv::Vec<T,2>> vec)
			: sqrt_Error_x{ sqrt_Err_x },
			sqrt_Error_y{ sqrt_Err_y },
			median_x{ media_x },
			median_y{ media_y },
			max_Error_x{ max_Err_x },
			max_Error_y{ max_Err_y },
			reproject_error_vec{std::move(vec)}
		{}
	};
	ReprojectionError(double, double, double, double, double, double, std::vector < cv::Vec<double, 2>>)->ReprojectionError<double>;
	ReprojectionError(double, double, double, double, double, double, std::vector < cv::Vec<float, 2>>)->ReprojectionError<float>;




	std::variant<ReprojectionError<float>, ReprojectionError<double>> m_repro_error{ ReprojectionError<double>() };
	std::variant<CalibrationPoints<float>, CalibrationPoints<double>> m_calib_points{ CalibrationPoints<double>() };



*/