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

	// Input[0]: const std::vector<cv::Mat> unwrap, unwrap.size() == 2, unwrap.type() == CV_64F for all images
	// Output: std::vector<cv::Mat> refined, refined.size() == 2
	std::vector<cv::Mat> refineUnwrap(
		const std::vector<cv::Mat>& unwrap,
		const int k_size_median = 5
	);
	
	// Input[0]: const std::vector<cv::Mat>& ref, ref.type() == CV_64F || CV_32F. [0]: Horizontal Reference, [1]: Vertical Reference
	// Input[1]: const cv::Size& sz, the size of the original sinus pattern in pix_x, pix_y
	// Input[2]: const double wavlength in pixels/2pi of the original sinus pattern-
	// Ouptput: std::vector<cv::Mat> ReferencePhase [0]: HorizontalReferencePhase, [1]: Vertical ReferencePhase
	// [0]: min val: 0; max val: pixel_x * wavelength[pixels/2pi]^-1 
	// [1]: min val: 0; max val: pixel_y * wavelength[pixels/2pi]^-1
	// |-> Of course only if the hole screen was visible in the images 
	std::vector<cv::Mat> prepareGrayCode_forUnwrap(
		const std::vector<cv::Mat>& ref,
		const cv::Size& sz,
		const double wavelength
	);
	// Input[0]: const std::vector<cv::Mat>& ref, ref.type() == CV_64F || CV_32F. [0]: Horizontal Reference, [1]: Vertical Reference
	// Input[1]: const cv::Size& sz, the size of the original sinus pattern in pix_x, pix_y
	// Input[2]: const double wavlength in pixels/2pi of the original sinus pattern-
	// Ouptput: std::vector<cv::Mat> ReferencePhase [0]: HorizontalReferencePhase, [1]: Vertical ReferencePhase
	// [0]: min val: 0; max val: pixel_x * wavelength[pixels/2pi]^-1 
	// [1]: min val: 0; max val: pixel_y * wavelength[pixels/2pi]^-1
	// |-> Of course only if the hole screen was visible in the images 
	std::vector<cv::Mat> prepareReferencePhase_forUnwrap(
		const std::vector<cv::Mat>& ref,
		const cv::Size& sz,
		const double wavelength
	);

	// Input
	// 1. image of type == CV_64F, 2. double max allowed value in image 3. min allwed value in image
	// Output cv::Mat of type cv_64F and size of img
	// The Method aborts if the Values in the image are higher than max val or lower than minval !
	cv::Mat quantizeImage(
		const cv::Mat& img,
		const double max_allowed = 255.0 + 1e-6,
		const double min_allowed = 0.0 - 1e-6
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
		const cv::Size sz,
		const std::string& path = std::string()
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
		const double wavelength,
		const int gridX,
		const int gridY,
		const double pixel_pitch_mm,
		bool flipunwrap_x_coordinates = false,
		bool flipunwrap_y_coordiantes = false
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
		const cv::Mat_<cv::Vec3d> display_coordiantes,
		cv::Mat_<cv::Vec3d> origin = cv::Mat_<cv::Vec3d>()
	);

	cv::Mat mapHitPointsToDisplayCoords(
		const cv::Mat& hitpoints,        // CV_64FC3, size = rays.size()
		const cv::Mat& display_points    // CV_64FC3, size = (H,W) of display raster
	);

	cv::Mat mean(const std::vector<cv::Mat>&);

	cv::Mat convertUnwrapToWorldCoord(
		const std::vector<cv::Mat>& unwrap,
		const double wavelength,
		const double pixelpitch
	);



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

