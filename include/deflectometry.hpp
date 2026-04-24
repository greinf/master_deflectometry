#ifndef	DEFLECTOMETRY_H
#define DEFLECTOMETRY_H
#include <memory>
#include <vector>
#include <exception>
#include <iostream>
#include <cassert>
#include <thread>
#include <opencv2/opencv.hpp>
#include "GrayCalibVector.hpp"
#include <numeric>
#include "utils.hpp"
#include <array>
#include "enums.hpp"
#include <optional>
#include <GrayCalibration_Utils.hpp>
#include "deflectometryUtils.hpp"

//Forward Decleration Enums + Class

enum class FrameRole;
enum class UnwrapMode;
enum class ReferenceMode;

class Pattern;
class AcquisitionWorker;
class ImageProcessing;
class ImageStore;
class ScreenDisplay;
class GrayCalibration;

namespace defl {
	class AcquisitionController;
	struct PhaseShiftConfig;
	struct CameraConfig;
}


class Deflectometry {
public:
	 Deflectometry();
	 ~Deflectometry();

	 cv::Mat getMask(
		 const std::vector<cv::Mat>&,
		 const double thresh,
		 const bool dilate
	 );

	 
	 // Method tries to acquiere config files if not available
	 // and saves at the selected Path in a XML file for each config. 
	 bool logging(
		 const std::string& paht);

	 // Methods tries to find Parallelogram corners of a Binary mask
	 // The corners are sorted in TL TR DL DR order! 
	 // The method can fail if more than 4 corners are found. 
	 // If happens: go into the cv::findcontours and change Epsilon. 
	 std::vector<std::pair<double,double>> findParallelogramCorners(const cv::Mat& bin);


	 std::vector<cv::Vec2d> getReferencePoint(
		 const std::vector<cv::Mat>& img,
		 const ReferenceMode mode,
		 const cv::Mat& mask
	 );

	 cv::Mat undistortImageManuell(
		 const cv::Mat& img,
		 const cv::Mat& cam,
		 const cv::Mat& dist_coeffs
	 );

	 GrayCalibVector borderPoints_gray_val(
		 const int pixel_x,
		 const int pixel_y,
		 const int sections_x = 0,
		 const int sections_y = 0
	 );

	 cv::Vec2d projectPoint_homography(
		 const cv::Vec2d& img,
		 const cv::Mat& homography
	 );

	 cv::Mat get_Homogrpahy_mat(
		 const std::vector < std::pair<int, int>>& src,
		 const std::vector < std::pair<double, double>>& dst,
		 const int method = 1
	 );

	 cv::Mat get_Homogrpahy_mat(
		 const std::vector<cv::Vec2d>& srcPoints,
		 const std::vector<cv::Vec2d>& dstPoints,
		 const int method = 1
	 );

	 // Method for manually acquiring images
	 // 1. In - FrameRole in this category the captured iamges are stored 
	 // 2. In - n_cameras specifies how many cameras should be used. 
	 // 3. Out - Vector of taken Picutres
	 // The method works by user Input. S for saving images and q for aborting
	 // While capturing a live image from the available Cameras is projected
	 std::vector<cv::Mat> getFrames(
		const FrameRole,
		const std::size_t n_cameras);

	 // Method for manually acquiring images
	 // 1. In - FrameRole in this category the captured iamges are stored 
	 // 2. In - n_cameras specifies how many cameras should be used. 
	 // 3. In - Given Picutre that can be potrayed on the Dispaly while capturing images. 
	 // 4. Out - Vector of taken Picutres
	 std::vector<cv::Mat> getFrames(
		 const FrameRole,
		 const std::size_t n_cameras,
		 const cv::Mat& img);

	 //1280 x 1024 IDS
	 std::array<double, (std::size_t)3> doWhiteBalance(
	 const std::vector<cv::Mat>&,
	 int roi_x = 250,
	 int roi_y = 250,
	 int roi_width = 200,
	 int roi_height = 200);


	 std::vector<cv::Vec2d> distortionPipelineTest(
	 const cv::Mat& mat,
	 const cv::Mat& dist,
	 const std::vector<cv::Vec2d>);
	 
	 // Method to Acquire the phase Shifted pictures. 
	 // Shift_mode: classifies wich shift mode is used 4 Shift mode or user defined
	 // n_pics: how many pictures per Phase picture
	 // save: Boolean value if the pictures should be saved. 
	 // number_of_Periods: Classfies the wavelength of the pattern by specifing the number
	 // of Waves on the y-axis
	 std::vector<cv::Mat> do_phase_measurement(Shift_mode,
		 const int n_pics,
		 const bool save,
		 const std::string& save_path,
		 const double number_of_perdios = 7,
		 _defl_::GrayCal::Method method = _defl_::GrayCal::Method::None,
		 const std::string& gray_calib_path = "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-15_GrayCalibSections"
	 );

	 std::vector<cv::Mat> do_wrapped_phase(
		 const std::vector<cv::Mat>& vec,
		 int n_pics_perPhase,
		 int n_shifts,
		 bool save,
		 const std::string& path_save);


	 // Input[0]: const std::vector<cv::Mat> img: contains the wrapped phase. Order depends on used method
	 //	|-> UnwrapMode::manually: Algorithm expects img.size() == 2 and img.type() == CV_64F for all images
	 //	|		Most basic method that only iterates over the wrapped phase and looks for jumps > pi
	 // |-> UnwrapMode::reference_Phase: Algorithm expects img.size() == 4 and img.type() == CV_64F for all images
	 // |		The Algorithm uses a reference phase ( A sin Wave with one Period is wrapped and used as reference for both directions)
	 // |		[0]: Horizontal Wrapped Phase, [1]: Horizontal ReferencePhase, [1]: Vertical Wrapped Phase, [2]: Vertical ReferencePhase
	 // |-> UnwrapMode::reference_GrayCode: Algorithm expects img.size() == 4 and img.type() == CV_64F for wrapped phase, img.type() == CV_32F || CV_64F for GrayCode
	 // |		The Algotihm uses a reference generated throgh the GrayCode decoding -> this images are normally in CV_32F
	 // |-> UnwrapMode::opencv Algotihm expect img.size() ==2 and img.type() == CV_64F || CV_32F for all images
	 // Input[1]: const cv::Mat& mask: mask.size() == img.size(), mask.type() == CV_8U
	 // Input[2]: See Input[0]
	 // Input[3]: bool if the value should be save to .xml file 
	 // Input[4]: const std::string& path, path for the output images
	 // Input[5]: const double wavelength: Parameter used for the Methods that use the referene Images
	 //			The wavelength must be given in pixels/2pi in Dispaly Pixels
	 // Input[6]: const cv::Size& sz: the size in pixelx, pixel_y of the original used Sinus pattern. 
	 // Ouput: std::vector<cv::Mat> img; where img.type() == CV_64FC1 and img.size() == input_img.size() for all images 
	 std::vector<cv::Mat> do_unwrapped_phase(
		 const std::vector<cv::Mat>& wrapped,
		 const cv::Mat& mask,
		 UnwrapMode mode,
		 bool save,
		 const std::string& save_path,
		 const double wavelength,
		 const cv::Size& disp_size = {1920,1080}
	 );

	 std::vector<cv::Mat> do_reprojection(
		 const std::vector<cv::Mat>& unwrapped,
		 const cv::Mat& mask, 
		 const cv::Mat& cam_Matrix,
		 const cv::Mat& dist_Coeffs,
		 const double wavelength,
		 const int grid_points_x,
		 const int grid_points_y,
		 const double pixel_pitch,
		 bool save, 
		 const std::string& save_path
	 );

	 cv::Mat generate_reference_Pattern(
		 const ReferenceMode mode,
		 const int n_pics,
		 bool save,
		 const std::string& path
	 );

	 // Method for testing implementation of GrayCalibration Class
	 void GrayCalibrationClassTest(
		 const _defl_::GrayCal::Method method,
		 const std::string path
	 );

	 // Function to create GrayValue calibration. 
	 // n_pics_per_value: the ammount of pictures taken per gray value
	 // save_path: The path where the LUT is stored as a .csv file
	 bool do_grayvalue_calibration(
		 const int n_pics_per_value,
		 const int n_steps, 
		 bool save,
		 const std::string& save_path,
		 const _defl_::GrayCal::Method method);

	 bool do_grayvalue_calibration(
		 const std::vector<cv::Mat>& gray_val,
		 cv::Mat& mask,
		 const int n_pics_per_value,
		 const int steps,
		 bool save,
		 const std::string& path,
		 const _defl_::GrayCal::Method method
	 );
	 

	 GrayCalibVector calc_response_curve_sections(
		 const int n_pics_per_value,
		 const int n_steps,
		 bool save,
		 const std::string& save_path,
		 const int sections_x,
		 const int sections_y
	 );

	 std::vector<cv::Mat> acquire_img(
		 const std::vector<cv::Mat>&,
		 const FrameRole dst,
		 const int n_pics_per_val
	 );

	 std::vector<cv::Mat> acquire_img(
		 const FrameRole src,
		 const FrameRole dst,
		 const int n_pics_per_value
	 );

	 std::vector<cv::Mat> do_camera_display_calibration(
		 const cv::Mat& cam_Mat,
		 const cv::Mat& distCoeffs,
		 const double point_distance,
		 const cv::Size pattern_size,
		 const Shift_mode mode,
		 const _defl_::GrayCal::Method method,
		 const std::string& calib_path,
		 const std::string& gray_calib_path,
		 const double pixel_pitch,
		 const double waves_per_y
	 );


	 std::vector<cv::Vec3d> extractValidVectorfromMat(
		 const cv::Mat_<cv::Vec3d>&,
		 const cv::Mat& mask
	 );

	 std::vector<cv::Vec3f> getCalibrationObjectPoints(
		 const cv::Size size,
		 const double dist
	 );


	 void load(FrameRole, const std::string&);

	 std::vector<cv::Mat> get(FrameRole role);

	 cv::Mat calcDistortionError(const cv::Mat& img);

	 cv::Mat generateCoordinateImg(
		 bool save,
		 const std::string& path);

	 // Does Setup the Pattern class. 
	 // Pattern class creates the Pattern like phase shift pattern and grayVal 
	 // Pattern class needs a shared Pointer to a img_store class
	 void setupPattern(
		 ImageStore& img_store,
		 int pixelX = 1920,
		 int pixelY = 1080);


	 void saveVecImage(
		 const FrameRole,
		 const std::vector<cv::Mat>&,
		 bool save,
		 const std::string& path
	 );

	 cv::Mat subtractSurface(
		 const cv::Mat& img,
		 const cv::Mat& mask,
		 bool show_subtracted = false
	 );

	 void saveSingleImage(
		 const FrameRole,
		 const cv::Mat&,
		 bool save,
		 const std::string& path);

	 std::vector<cv::Mat> generatePattern(
		 const Shift_mode mode,
		 const _defl_::GrayCal::Method method,
		 const std::string& calib_path,
		 const FrameRole dst,
		 bool save, 
		 const std::string& path,
		 double n_periods_in_y = 10);

	 
	 std::vector<cv::Mat> generateCartesian(
		 bool save,
		 const std::string& path,
		 int gridX,
		 int gridY);

	 std::optional<cv::Point3d> triangulateFromPixels(
		 const cv::Point2d& px1,
		 const cv::Point2d& px2,
		 const cv::Mat& K1, const cv::Mat& D1,
		 const cv::Mat& K2, const cv::Mat& D2,
		 const cv::Mat& R, const cv::Mat& t
	 );

	 std::vector<cv::Mat> testReprojection(
		 const std::string& camMatrix_path,
		 const std::string& calibPath,
		 const std::string& savePath,
		 Shift_mode shiftmode,
		 _defl_::GrayCal::Method calmethod,
		 double n_perios_in_y,
		 bool synthetic_images,
		 UnwrapMode unwrap,
		 bool useIndistort
	 );



	 static cv::Point2d detectLaserSpotSubpix(
		 const cv::Mat& imgOn, const cv::Mat& imgOff,
		 double threshFrac = 0.4,   // threshold relativ zum Max in diff
		 int morphIters = 1         // cleanup
	 ) 
	 {
		 CV_Assert(!imgOn.empty() && !imgOff.empty());
		 CV_Assert(imgOn.size() == imgOff.size());

		 cv::Mat onGray, offGray;
		 if (imgOn.channels() == 3) cv::cvtColor(imgOn, onGray, cv::COLOR_BGR2GRAY);
		 else onGray = imgOn;
		 if (imgOff.channels() == 3) cv::cvtColor(imgOff, offGray, cv::COLOR_BGR2GRAY);
		 else offGray = imgOff;

		 // diff = on - off (saturating)
		 cv::Mat diff;
		 cv::subtract(onGray, offGray, diff, cv::noArray(), CV_16S);
		 diff = cv::abs(diff);

		 // Smooth a bit (helps centroid stability)
		 cv::Mat diffBlur;
		 cv::GaussianBlur(diff, diffBlur, cv::Size(0, 0), 1.0);

		 // Normalize to 8-bit for thresholding
		 double minV, maxV;
		 cv::minMaxLoc(diffBlur, &minV, &maxV);
		 if (maxV < 1.0)
			 throw std::runtime_error("Laser spot not detectable (diff too small).");

		 cv::Mat diff8;
		 diffBlur.convertTo(diff8, CV_8U, 255.0 / maxV);

		 // Threshold around brightest region
		 const int thr = static_cast<int>(std::round(threshFrac * 255.0));
		 cv::Mat mask;
		 cv::threshold(diff8, mask, thr, 255, cv::THRESH_BINARY);

		 // Morph cleanup
		 if (morphIters > 0) {
			 cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
			 cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), morphIters);
			 cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), morphIters);
		 }

		 // Keep largest blob (laser spot)
		 std::vector<std::vector<cv::Point>> contours;
		 cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		 if (contours.empty())
			 throw std::runtime_error("No blob found for laser spot.");

		 int bestIdx = -1;
		 double bestArea = 0.0;
		 for (int i = 0; i < (int)contours.size(); ++i) {
			 double a = cv::contourArea(contours[i]);
			 if (a > bestArea) { bestArea = a; bestIdx = i; }
		 }
		 if (bestIdx < 0 || bestArea < 1.0)
			 throw std::runtime_error("Laser blob too small / not stable.");

		 cv::Mat spotMask = cv::Mat::zeros(mask.size(), CV_8U);
		 cv::drawContours(spotMask, contours, bestIdx, cv::Scalar(255), cv::FILLED);

		 // Weighted centroid using diff intensity (gives subpixel-ish stability)
		 cv::Mat weighted;
		 diff8.copyTo(weighted, spotMask);   // Maske anwenden

		 cv::Moments m = cv::moments(weighted, false);
		 if (std::abs(m.m00) < 1e-9)
			 throw std::runtime_error("Moments failed for laser spot.");

		 return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
	 }


	 // --- 2) Triangulation: 2D->3D aus Stereo-Kalibrierung ---
	 static cv::Point3d triangulatePointStereo(
		 const cv::Point2d& p1, const cv::Point2d& p2,
		 const cv::Mat& K1, const cv::Mat& D1,
		 const cv::Mat& K2, const cv::Mat& D2,
		 const cv::Mat& R, const cv::Mat& T
	 ) {
		 // Undistort points to normalized camera coordinates
		 std::vector<cv::Point2f> v1{ cv::Point2f((float)p1.x, (float)p1.y) };
		 std::vector<cv::Point2f> v2{ cv::Point2f((float)p2.x, (float)p2.y) };

		 std::vector<cv::Point2f> u1, u2;
		 cv::undistortPoints(v1, u1, K1, D1); // -> normalized (x,y) with z=1
		 cv::undistortPoints(v2, u2, K2, D2);

		 // Projection matrices in normalized coords:
		 // P1 = [I|0]
		 // P2 = [R|T]
		 cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
		 cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
		 cv::Mat I = cv::Mat::eye(3, 3, CV_64F);

		 I.copyTo(P1(cv::Rect(0, 0, 3, 3)));
		 // last col of P1 is 0

		 R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
		 T.copyTo(P2(cv::Rect(3, 0, 1, 3)));

		 cv::Mat X4;
		 cv::triangulatePoints(P1, P2, u1, u2, X4); // 4xN

		 

		 // homogeneous -> Euclidean
		 const double w = X4.at<float>(3, 0);
		 if (std::abs(w) < 1e-12)
			 throw std::runtime_error("Triangulation produced invalid homogeneous coord.");

		 cv::Point3d X(
			 X4.at<float>(0, 0) / w,
			 X4.at<float>(1, 0) / w,
			 X4.at<float>(2, 0) / w
		 );

		 return X; // in camera1 coordinate system
	 }
	

	 // --- 3) Gesamt: 4 Bilder -> Entfernung ---
	 struct LaserDistanceResult {
		 cv::Point2d p1_px, p2_px;   // detected pixel positions
		 cv::Point3d X_cam1;         // 3D point in cam1 coords
		 double distance_cam1;       // ||X||
	 };

	 LaserDistanceResult computeLaserDistanceFrom4Images(
		 const cv::Mat& cam1_on, const cv::Mat& cam1_off,
		 const cv::Mat& cam2_on, const cv::Mat& cam2_off,
		 const cv::Mat& K1, const cv::Mat& D1,
		 const cv::Mat& K2, const cv::Mat& D2,
		 const cv::Mat& R, const cv::Mat& T
	 ) {
		 LaserDistanceResult res;

		 res.p1_px = detectLaserSpotSubpix(cam1_on, cam1_off);
		 res.p2_px = detectLaserSpotSubpix(cam2_on, cam2_off);

		 res.X_cam1 = triangulatePointStereo(res.p1_px, res.p2_px, K1, D1, K2, D2, R, T);
		 res.distance_cam1 = std::sqrt(res.X_cam1.x * res.X_cam1.x + res.X_cam1.y * res.X_cam1.y + res.X_cam1.z * res.X_cam1.z);

		 return res;
	 }

	 // Input[0]: Enum for GrayCalibration Method
	 // Input[1]: const std::string& to the Calibration folder in xml
	 // Output: bool Value if Setup was succesfull
	 bool setupCalibration(
		 const _defl_::GrayCal::Method methode,
		 const std::string& path
	 );

	 cv::Mat applyCalibration(
		 const cv::Mat& image,
		 cv::Mat& mask,
		 const _defl_::GrayCal::Method methode
	 );

	 // calculates the element wise difference of a two images and return the result. 
	 // when save value is set to true the value is saved in Debug FrameRole of the ImageStore
	 cv::Mat get_difference_debug(const cv::Mat& mat1, const cv::Mat& mat2, bool save = false); 

	 cv::Mat undistortImage(const cv::Mat&, const cv::Mat& cam_Matrix, const cv::Mat& dist_coeffs);

	 cv::Mat distortImage_manual(const cv::Mat& img, const cv::Mat& cam_Matrix, const cv::Mat& dist_coeffs);

	 // This should in Future be use instead of the many forwarding functions to image Processing
	 // Sadly this has to be done at another time 
	 ImageProcessing& processing() const {
		 return *m_img_processing;
	 };

	 // Same here for the pattern class but this should be also done at another time. 
	 Pattern& img_generation() const {
		 return *m_pattern;
	 }

	 ImageStore& image_store() const {
		 return *m_img_store;
	 }

	 
	 std::unique_ptr<ImageStore> m_img_store{ nullptr }; // Must be shared_ptr
	 std::unique_ptr<Pattern> m_pattern{ nullptr }; // could be Unique
	 std::unique_ptr<AcquisitionWorker> m_acquisition_worker{ nullptr };  // Must be shared
	 std::unique_ptr<ImageProcessing> m_img_processing{ nullptr };  // could be unqiue
	 std::unique_ptr<ScreenDisplay> m_screenDisplay{ nullptr };  // Better shared 
	 std::unique_ptr<defl::AcquisitionController> m_acquisition_controller{ nullptr };  // better shared
	 std::unique_ptr<GrayCalibration> m_calibration{ nullptr };


private:
	void show_norm(const std::vector<cv::Mat>& images, const std::string& name) {
		for (const auto& imga : images) {
			cv::Mat img;
			cv::normalize(imga, img, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::namedWindow(name, cv::WINDOW_NORMAL);
			cv::imshow(name, img);
			cv::waitKey(0);
			cv::destroyWindow(name);
		}
	}
	

	// Input[0]: enum to decide which method to use -> located in deflectometryUtils.hpp
	// Input[1]: const std::vector of 256 pictures showing the gray values. 1 channel, type: double
	// Input[2]: mask for showing the valid pixels in the picture 
	// Input[3]: path where to the save folder to save gray Calibration
	// Ouput: Returns bool if calibration was succesfull
	bool do_grayvalue_calibrationTypeDispatch(
		const _defl_::GrayCal::Method method,
		const std::vector<cv::Mat>& gray_img,
		cv::Mat& mask,
		const std::string& path);


	// If Curly brackets for default initialization are used forward decleration breaks
	std::vector<std::unique_ptr<defl::PhaseShiftConfig>> m_pattern_config;
	std::vector<std::unique_ptr<defl::CameraConfig>> m_camera_config;

	std::pair<double, double> fitLine1D(const std::vector<double>& y);

	// Methods to connect and Setup Hardware -> Camera and Acquisitionworker class get Setup
	// Set camera_n to 1 if only one camera is used. 2 For Two cameras at a time. 
	bool init(std::size_t camera_n);

	// Class to disconnect from Hardware -> Camera and Acquisitionworker get set to nullptr
	bool disconnect();

	bool check_synthaticall_points(const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>>&);
	// --- End -----


	void saveSliceToCSV(const std::string& filename,
		const std::vector<double>& unwrap,
		const std::pair<double, double>& unwrapFit,// regressions a,b unwrap
		const std::vector<double>& repro,
		const std::pair<double, double>& reproFit);

};


#endif 