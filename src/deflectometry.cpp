#include "deflectometry.hpp"
#include "screen.hpp"
#include "acquisitionworker.hpp"
#include "cassert"
#include "enums.hpp"
#include "imgProcessing.hpp"
#include "camera_calib.hpp"
#include <fstream>
#include <filesystem>
#include <utility>
#include "imageStore.hpp"
#include "ScreenDisplay.hpp"
#include "runtime/AcquisitionController.hpp"
#include "screen.hpp"
#include "config/CameraConfig.hpp"
#include "config/PhaseShiftConfig.hpp"
#include "algorithm"
#include "CameraNew.hpp"
// Response in Section
#include "GrayCalibVector.hpp"
// Gray Calib class
#include "GrayCalibration.hpp"
#include "RowPolicy.hpp"


// ****Just for Fun ****
//Marcro to check type of expression at compile time 
// (type*)0 creates a null pointer of type 'type*', &expr gets the address of expr
// If expr is not of type 'type', the comparison will fail, causing a compile-time error.
// C++ forbits comparing two pointer of incompatible types. Therefor this will create a compile time error if types do not match.
// The literal 0 gets cast to a pointer of type 'type*', this allows the comparison. 
#define CHECK_TYPE(expr, type) ((void)((type*)0 == &(expr))))

/*
 
// Very cool demonstratoin of perfect forwarding. 
// Differences between compile time values and runtime values must be understood. 

#include <iostream>
#include <string>
#include <utility>   // for std::forward

// our wrapper template
template<typename F, typename... Args>
auto log_call(F&& f, Args&&... args)
{
	std::cout << "Calling function...\n";
	// perfectly forward f and all args
	return std::forward<F>(f)(std::forward<Args>(args)...);
}

// --- a few functions to test with ---

void greet(const std::string& name)
{
	std::cout << "Hello, " << name << "!\n";
}

void modify(std::string& s)
{
	s += " (modified)";
}

void print_ref(std::string&& s)
{
	std::cout << "Rvalue received: " << s << '\n';
}

int sum(int a, int b)
{
	return a + b;
}

//Access datatype of N-th argument
* By putting the compile time variables through std::forward_as_tuple, we create a tuple of references that preserve the value category (lvalue/rvalue) of each argument.
* This can be accessed. The compiler itself just uses datatypes. Through the usage of a runtime variable, we can access it. 
* Also very intersting: fold expression ((std::cout << args << ' '), ...); // fold expression. This is opened by the compiler to print all arguments.
* ((expression), ...) -> (expr op ...) expands to -> (((expr1 op expr2) op expr3) op ...)
* 
template<std::size_t N, typename... Ts>
decltype(auto) get_arg(Ts&&... ts)
{
	return std::get<N>(std::forward_as_tuple(std::forward<Ts>(ts)...));
}

int main()
{
	std::string name = "Guido";

	//Call with a normal lvalue reference
	log_call(greet, name);

	//Call with an lvalue reference that gets modified
	log_call(modify, name);
	std::cout << "After modify: " << name << '\n';

	//Call with an rvalue
	log_call(print_ref, std::string("temporary"));

	//Call with a function returning a value
	int result = log_call(sum, 10, 20);
	std::cout << "Sum result: " << result << '\n';

	//Call with a lambda
	log_call([](auto&& msg) { std::cout << "Lambda says: " << msg << '\n'; }, "hi");
}


// std::apply is more or less the inverse of std::forward_as_tuple. A small simplified example here:

template<typename F, typename Tuple, std::size_t... I>
decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
{
	// unpack tuple elements into the function call
	return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

template<typename F, typename Tuple>
decltype(auto) apply(F&& f, Tuple&& t)
{
	constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;
	return apply_impl(
		std::forward<F>(f),
		std::forward<Tuple>(t),
		std::make_index_sequence<N>{} // generates 0,1,2,...,N-1
	);
}

*/
cv::Mat rotImage180(const cv::Mat&);

cv::Mat showRawMaxValred(const cv::Mat& mat) {
	cv::Mat color;
	cv::Mat mask(mat > 250);
	cv::cvtColor(mat, color, cv::COLOR_GRAY2BGR);
	color.setTo(cv::Scalar(0, 0, 255), mask);
	return color;
}


Deflectometry::Deflectometry() {
	m_img_store = std::make_unique<ImageStore>();
	// For now imageProcessing Strays in constructor
	m_img_processing = std::make_unique<ImageProcessing>(*m_img_store);
	m_screenDisplay = std::make_unique<ScreenDisplay>();
	
}

Deflectometry::~Deflectometry() = default;

auto nearly_equal = [&](double a, double b) {
	double eps = 1e-9;
	return std::abs(a - b) < eps;
	};


bool Deflectometry::check_synthaticall_points(const std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>>& pts) {
	CV_Assert(pts.first.size() == pts.second.size());
	int counter{ 0 };
	for (std::size_t i = 0; i < pts.first.size(); ++i) {
		if (!nearly_equal(pts.first[i][0], pts.second[i][0]) ||
			!nearly_equal(pts.first[i][1], pts.second[i][1]))
		{
			std::cout << "ImagePoints: " << pts.first[i] << '\n';
			std::cout << "ObjectPoints: " << pts.second[i] << '\n';
			++counter;
		}
	}
	return (counter > 0) ? (false) : (true);
}

std::vector<cv::Mat> Deflectometry::generateCartesian(
	bool save,
	const std::string& path,
	int gridX,
	int gridY)
{
	CV_Assert((gridX > 0) && (gridY > 0));
	CV_Assert(!path.empty());
	setupPattern(*m_img_store);

	setupCalibration(CalibrationMethod::None, "");

	cv::Mat cartesian = m_pattern->generateCartesian(gridX, gridY);

	m_img_store->add(FrameRole::GridPattern, cartesian);
	if (save) {
		m_img_store->saveRole(FrameRole::GridPattern, path);
		m_img_store->saveRoleXML(FrameRole::GridPattern, path);
	}
	m_img_store->show(FrameRole::GridPattern);
	std::vector<cv::Mat> cartesianvec{ cartesian };
	return cartesianvec;
}

cv::Mat Deflectometry::generateCoordinateImg(bool save, const std::string& path) {
	setupPattern(*m_img_store);

	setupCalibration(CalibrationMethod::None, "");

	cv::Mat coordinateImg = m_pattern->generatecoordianteImg();
	m_img_store->add(FrameRole::DistortionCalib, coordinateImg);
	if (save) {
		//m_img_store->saveRole(FrameRole::DistortionCalib, path);
		m_img_store->saveRoleXML(FrameRole::DistortionCalib, path);
	}
	//m_img_store->show(FrameRole::DistortionCalib);

	return coordinateImg;
}

std::vector<cv::Vec2d> Deflectometry::distortionPipelineTest(
	const cv::Mat& mat,
	const cv::Mat& dist,
	const std::vector<cv::Vec2d> img_pts)
{
	CV_Assert(!img_pts.empty());
	CV_Assert(mat.size() == cv::Size(3, 3));
	
	std::vector<cv::Vec2d> distorted;
	for (const auto& vec : img_pts) {
		distorted.push_back(m_img_processing->newtonSolverdistort(vec, mat, dist));
	}

	std::vector<cv::Vec2d> distorted1;
	cv::undistortPoints(
		img_pts,
		distorted1,
		mat,
		dist,
		cv::noArray(),
		mat
	);

	std::vector<cv::Vec2d> undistorted;
	for (const auto& vec : distorted) {
		undistorted.push_back(m_img_processing->undistortImagePts(vec, mat, dist));
	}


	double sum_sq = 0.0;
	double max_err = 0.0;

	std::vector<cv::Vec2d> error;
	error.reserve(img_pts.size());

	for (size_t i = 0; i < img_pts.size(); ++i) {
		cv::Vec2d e = img_pts[i] - undistorted[i];
		error.push_back(e);

		double err = cv::norm(e);          // ||e||
		sum_sq += err * err;               // sum ||e||^2
		max_err = std::max(max_err, err);  // max ||e||
	}

	double l2_total = std::sqrt(sum_sq);                      // sqrt(sum ||e||^2)
	double rmse = std::sqrt(sum_sq / img_pts.size());     // sqrt(mean ||e||^2)

	std::cout << "Max Err:  " << max_err << '\n';
	std::cout << "L2 total: " << l2_total << '\n';
	std::cout << "RMSE:     " << rmse << '\n';

	return error;
}

std::vector<cv::Mat> Deflectometry::do_reprojection(
	const std::vector<cv::Mat>& unwrapped,
	const cv::Mat& mask,
	const cv::Vec2d refPoint,
	const cv::Mat& cam_Matrix,
	const cv::Mat& dist_Coeffs,
	const double wavelength,
	const int grid_points_x,
	const int grid_points_y,
	const double pixel_pitch,
	bool save,
	const std::string& save_path,
	const double screen_width,
	const double screen_height) 
{

	// --- Set to 0 for debugging ---
	//cv::Mat mask = m_img_processing->createMask(unwrapped, 0.0);

	std::pair<std::vector<cv::Vec2d>, std::vector<cv::Vec3d>> calibrationPoints =
		m_img_processing->do_calibration_Points(
			unwrapped,
			mask,
			refPoint,
			wavelength,
			grid_points_x,
			grid_points_y,
			pixel_pitch,
			screen_width,
			screen_height);

	// Only valid check for pixel_pitch = 1
	//if (check_synthaticall_points(calibrationPoints)) std::cout << "Happy Calibration Points ";
	//else std::cout << "Not so happy ";
	// --- For debugging --
	/*cv::Mat synthetical_camera_matrix = cv::Mat::eye(3, 3, CV_64F); 
	cv::Mat synthetical_distortion = cv::Mat::zeros(dist_Coeffs[0].size(), CV_64F);*/
	// --- end ---

	cv::Mat rvec, tvec;
	bool ok = cv::solvePnP(calibrationPoints.second, calibrationPoints.first,
		cam_Matrix, dist_Coeffs, rvec, tvec,
		false,
		cv::SOLVEPNP_ITERATIVE);
	if (!ok) throw std::runtime_error("solvePnP failed.");
	std::cout << "Translation vec " << cv::norm(tvec) << '\n';

	std::cout << "Rotation Vector " << rvec << '\n';
	
	std::vector<cv::Point2d> img_proj;
	cv::projectPoints(
		calibrationPoints.second, // objectPoints
		rvec, tvec,
		cam_Matrix,
		dist_Coeffs,
		img_proj
	);

	// jetzt Pixel-Reprojection-Error
	double err = 0;
	for (size_t i = 0; i < img_proj.size(); ++i) {
		err += cv::norm(img_proj[i] - cv::Point2d(calibrationPoints.first[i]));
	}
	err /= img_proj.size();
	std::cout << "Mean pixel reprojection error: " << err << std::endl;

	double sqrt_err, max_err;

	cv::Mat reprojection_img = m_img_processing->
		do_reprojection_error(
			calibrationPoints, 
			cam_Matrix,
			dist_Coeffs,
			rvec,
			tvec,
			&sqrt_err,
			&max_err,
			mask);

	std::cout << "Sqrt Error: " << sqrt_err << '\n'
		<< "Max Error: " << max_err << '\n';
	std::vector<cv::Mat> xy(2);
	cv::split(reprojection_img, xy);
	m_img_store->add(FrameRole::ReprojectionX, xy[0]);
	m_img_store->add(FrameRole::ReprojectionY, xy[1]);
	m_img_store->show(FrameRole::ReprojectionX);
	m_img_store->show(FrameRole::ReprojectionY);

	//m_img_store->saveRoleXML(FrameRole::ReprojectionX, )
	if (save) {
		m_img_store->saveRoleXML(FrameRole::ReprojectionX, save_path);
		m_img_store->saveRoleXML(FrameRole::ReprojectionY, save_path);
	}
	return xy;
}


// ReferenceMode cross = 0,
// checkerboard = 1
cv::Mat Deflectometry::generate_reference_Pattern(
	const ReferenceMode mode,
	const int n_pics,
	bool save,
	const std::string& path) 
{
	CV_Assert(n_pics > 0);
	CV_Assert(static_cast<int>(mode) < 2 &&
		static_cast<int>(mode) >= 0);

	//cv::Mat pattern = m_pattern->generateCross(1920, 1080, 1920.0 / 2, 1080.0 / 2, 4);

	cv::Mat pattern = (static_cast<int>(mode)) ?
		m_pattern -> generateCheckerboard(1920, 1080, 0) :
		m_pattern -> generateCross(1920, 1080, 1920.0 / 2, 1080.0 / 2, 4);
	return pattern;

}

// ReferenceMode cross = 0,
// checkerboard = 1
// Only the Checkerboard method is fully functioning. 
// Two refinement methods are implemented. One Template matching optimizer doTemplateMatching()
// and one method as a rebuild of the openCv cornersubPix that allows working with double values. 

std::vector<cv::Vec2d> Deflectometry::getReferencePoint(
	const std::vector<cv::Mat>& img,
	const ReferenceMode mode,
	const cv::Mat& mask)
{
	CV_Assert(!img.empty());
	
	std::vector<cv::Vec2d> refPoint;
	std::vector<cv::Vec2d> refinedPoint;
	switch (static_cast<int>(mode)) {
	case(static_cast<int>(ReferenceMode::cross)):
	{
		std::cout << "Not implemented Path !!";
		return {};
	}
	case(static_cast<int>(ReferenceMode::checkerboard)):
	{
		// --- Start with first detecting rough position of possible corners ---
		refPoint = m_img_processing->harrisCornerDetection(img, mask, { 5,5 });

		// --- Template Matching method ---
		// cv::Mat templ = m_img_processing->getTemplateChess(img[0], cv::Size{21, 21});
		// refinedPoint = m_img_processing->doTemplateMatching(refPoint, img, templ);

		// --- Open Cv method ---
		// needs convertion to cv::Point2f in floating values
		/*cv::Mat templfloat;
		cv::Mat img8u;
		img[0].convertTo(img8u, CV_8U);
		std::vector<cv::Point2f> cornersub;
		cornersub.emplace_back(cv::Point2f{ static_cast<float>(refPoint[0][1]),
			static_cast<float>(refPoint[0][0]) });
		cv::cornerSubPix(img8u, cornersub, { 11,11 }, { -1,-1 }, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.000001));*/

		// --- Own Implementatoin of refine Croner ---
		// methods works directly with cv::Vec2d (y,x) 
		std::vector<cv::Vec2d> refinedCorner =
			m_img_processing->refineCorner(img, refPoint, { 21,21 }, 1E-10);


		cv::Mat color;
		cv::Mat check;
		cv::normalize(img[0], check, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::cvtColor(check, color, cv::COLOR_GRAY2BGR);

		cv::Point point(
			static_cast<int>(refinedCorner[0][1]),
			static_cast<int>(refinedCorner[0][0]));
		cv::drawMarker(color, point, cv::Scalar(255, 0, 0), 0, 100);

		cv::imshow("checking", color);
		cv::waitKey(0);

		return refinedCorner;
	}
	default:
	{
		std::cout << "Invalid Value for refine corner. Return empty Vector \n";
		return {};
	}
	}
}



cv::Mat Deflectometry::undistortImage(const cv::Mat& img, const cv::Mat& cam_Matrix, const cv::Mat& dist_Coeffs) {

	cv::Size imageSize = img.size();
	//cv::Mat distorted = m_img_processing->distortImage(img, cam_Matrix, dist_Coeffs);
	cv::Mat distorted = m_img_processing->undistortImage(img, cam_Matrix, dist_Coeffs);
	cv::Mat distorted32;
	distorted.convertTo(distorted32, CV_32F);

	//cv::undistort(img, distorted, cam_Matrix, dist_Coeffs, cv::noArray());
	//m_img_store->add(FrameRole::Debug, distorted);
	cv::Mat undistort, undistort1;
	
	cv::Mat map1, map2;
	
	cv::initUndistortRectifyMap(
		cam_Matrix, dist_Coeffs, Mat(),
		cam_Matrix, imageSize,
		CV_32FC2, map1, map2);

	cv::remap(img, undistort1, map1, map2, cv::INTER_CUBIC, cv::BORDER_CONSTANT);

	cv::Mat undistorted64;
	undistort1.convertTo(undistorted64, CV_64F);

	m_img_store->add(FrameRole::Debug, distorted32);

	m_img_store->add(FrameRole::Debug, undistort1);
	//m_img_store->show(FrameRole::Pattern);
	//m_img_store->show(FrameRole::Debug);
	return undistorted64;
}


cv::Mat Deflectometry::undistortImageManuell(
	const cv::Mat& img,
	const cv::Mat& cam,
	const cv::Mat& dist_coeffs)
{
	CV_Assert(!img.empty());
	CV_Assert(cam.size() == cv::Size(3, 3));

	cv::Mat undistorted = m_img_processing->undistortImage(img, cam, dist_coeffs);
	return undistorted;
}



cv::Mat Deflectometry::calcDistortionError(const cv::Mat& img) {
	cv::Mat error = 
		m_img_processing->calcDistortionError(img);
	return error;
}

void Deflectometry::saveVecImage(
	const FrameRole role,
	const std::vector<cv::Mat>& vec,
	bool save,
	const std::string& path)
{
	CV_Assert(!vec.empty());
	for (const auto& img : vec) {
		saveSingleImage(role, img, save, path);
	}
}


void Deflectometry::saveSingleImage(
	const FrameRole role,
	const cv::Mat& img,
	bool save,
	const std::string& path) 
{
	CV_Assert(img.size().area() > 0);
	CV_Assert(img.channels() <= 2);

	if(img.channels()==2){
		std::cout << "Save Distortion Error Pics, Special case \n";
		cv::Mat arr[2];
		cv::split(img, arr);
		m_img_store->add(FrameRole::DistortErrX, arr[0]);
		m_img_store->add(FrameRole::DistortErrY, arr[1]);
		if (save) {
			m_img_store->saveRoleXML(FrameRole::DistortErrX, path);
			m_img_store->saveRoleXML(FrameRole::DistortErrY, path);
		}
		return;
	}

	m_img_store->add(role, img);
	if (save) {
		m_img_store->saveRole(role, path);
		m_img_store->saveRoleXML(role, path);
	}
}

cv::Mat Deflectometry::distortImage_manual(
	const cv::Mat& img,
	const cv::Mat& cam_Matrix,
	const cv::Mat& dist_Coeffs)
{
	CV_Assert(img.rows > 2 && img.cols > 2);
	CV_Assert(dist_Coeffs.rows > 0);
	CV_Assert(cam_Matrix.rows == 3 && cam_Matrix.cols == 3);
	cv::Mat mapx(img.size(), CV_64F, cv::Scalar(0)), mapy(img.size(), CV_64F, cv::Scalar(0));

	for (int row = 0; row < img.rows; ++row) {
		double* ptr_x = mapx.ptr<double>(row);
		double* ptr_y = mapy.ptr<double>(row);
		for (int cols = 0; cols < img.cols; ++cols) {
			//cv::Vec2d distorted = newtonSolver(cv::Vec2d(row, cols), )
			cv::Vec2d distorted_coordinates =
				m_img_processing->newtonSolverdistort(cv::Vec2d(cols, row), cam_Matrix, dist_Coeffs);
			ptr_x[cols] = distorted_coordinates[0];
			ptr_y[cols] = distorted_coordinates[1];
		}
	}

	cv::Mat distorted;
	cv::Mat img32, mapx32, mapy32;
	img.convertTo(img32, CV_32F);
	mapx.convertTo(mapx32, CV_32F);
	mapy.convertTo(mapy32, CV_32F);

	cv::remap(img32, distorted, mapx32, mapy32, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

	cv::Mat distorted64;
	
	distorted.convertTo(distorted64, CV_64F);
	/*m_img_store->add(FrameRole::Debug, distorted);
	m_img_store->show(FrameRole::Debug);*/
	
	// If multiple Channels are there. 
	/*std::vector<cv::Mat> split_xy;
	cv::split(distorted64, split_xy);
	for (const auto& img : split_xy) { m_img_store->add(FrameRole::Debug, img); }
	m_img_store->show(FrameRole::Debug);*/
	return distorted64;
}


cv::Mat Deflectometry::get_difference_debug(const cv::Mat& mat1, const cv::Mat& mat2, bool save)
{
	cv::Mat difference = mat1 - mat2;
	if (save) {
		m_img_store->add(FrameRole::Debug, difference);
		m_img_store->show(FrameRole::Debug);
	}
	return difference;
}

bool Deflectometry::init() {
	try {
		// This should be at one time changed to unqiue Ptr
		std::shared_ptr<CameraN> m_camera = std::make_shared<CameraN>(CameraN::Backend::VIMBA);

		m_camera->open();
		
		m_acquisition_controller = std::make_unique<defl::AcquisitionController>();
		
		m_acquisition_controller->acquisition_active = m_camera->isRunning();
		if (!m_acquisition_controller->acquisition_active) throw std::runtime_error("Camera Acuqisition not active \n");

		if (!m_camera_config.empty()) throw std::runtime_error("A camera Config is already loaded \n");

		for (const auto& cfg : m_camera->getCamConfig()) {
			m_camera_config.push_back(std::make_unique<defl::CameraConfig>(cfg));
		}
		
		m_acquisition_worker = std::make_unique<AcquisitionWorker>(
			std::move(m_camera), // moves pointer pointer to Acquisitionworker
			*m_img_store,
			*m_acquisition_controller,
			*m_screenDisplay
			);
		return true;
	}
	catch (std::exception& e) { std::cout << "EXCEPTION: " << e.what() << std::endl; return false; }
}

cv::Mat Deflectometry::getMask(
	const std::vector<cv::Mat>& contrast,
	const double thresh,
	const bool dilate)
{
	CV_Assert(!contrast.empty());
	CV_Assert(thresh >= 0);
	
	return m_img_processing->createMask(contrast, thresh, dilate);

}

bool Deflectometry::disconnect() {
	try {
		m_acquisition_worker.reset();
		m_acquisition_worker = nullptr;
		m_acquisition_controller.reset();
		m_acquisition_controller = nullptr;
	    
		
		return true;
	}
	catch (std::exception& e) { std::cout << "EXCEPTION: " << e.what() << std::endl; return false; }
}

std::vector<cv::Mat> Deflectometry::do_unwrapped_phase(
	const std::vector<cv::Mat>& wrapped_Phase,
	const cv::Mat& mask,
	UnwrapMode mode,
	bool save,
	const std::string& save_path)
{
	CV_Assert(wrapped_Phase[0].size() == mask.size());
	//CV_Assert(wrapped_Phase[0].type() == mask.type());
	std::vector<cv::Mat> unwrappedPhase(2);
	if (mode == UnwrapMode::manually) {
		unwrappedPhase = 
			m_img_processing -> manual_phaseUnwrap(wrapped_Phase, mask);
	}
	else if (mode == UnwrapMode::opencv) {
		unwrappedPhase =
			m_img_processing->unwrapped_phase(wrapped_Phase, mask);
	}

	for (const auto& img : unwrappedPhase) { m_img_store->add(FrameRole::UnwrappedPhase, img); }

	if (save) {
		m_img_store->saveRole(FrameRole::UnwrappedPhase, save_path);
		m_img_store->saveRoleXML(FrameRole::UnwrappedPhase, save_path);
	}
	m_img_store->show(FrameRole::UnwrappedPhase);

	return unwrappedPhase;
}


std::vector<cv::Mat> Deflectometry::createSyntheticalImages(
	const double gamma,
	const bool luminance,
	const double image_height,  // Mirror circumference  
	const int dest_width,      // Mako G-507-B width
	const int dest_height,     // Mako G-507-B height
	const double dispaly_pixel_pitch,
	const Shift_mode mode,
	const CalibrationMethod method,
	const int display_pixel_x,
	const int display_pixel_y,
	const int n_periods_in_y,
	const double aperture_number,
	const double distance,     // f = 1600 distance 2*f
	const double object_height, // 2/3" Sensor 8,8 * 6,6 
	const RoiBorders<double> homography_points)
{
	CV_Assert(display_pixel_x > 0 && display_pixel_y > 0);
	CV_Assert(aperture_number >= 0 && n_periods_in_y > 0);
	CV_Assert(distance >= 0 && object_height > 0);
	CV_Assert(image_height >= 0);
	CV_Assert(dest_width >= 0 && dest_height >= 0);
	CV_Assert(gamma >= 0);

	setupPattern(*m_img_store);

	std::vector<cv::Mat> pattern =
		m_pattern->generate_phaseShift(
			mode,
			10,
			127.5,
			127.5,
			1920,
			1080,
			UniformRowsCols{});

	/*for (const auto& imga : pattern) {
		cv::Mat img;
		cv::normalize(imga, img, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("pattern", img);
		cv::waitKey(0);
		cv::destroyWindow("pattern");
	}
	*/

	std::vector<cv::Mat> pattern_gamma;
	if (gamma) {
		for (const auto& img : pattern) {
			pattern_gamma.emplace_back(
				m_img_processing->do_gamma_distortion(2, img, UniformRowsCols{}));
		}
	}

	else pattern_gamma = pattern;

	
	/*for (const auto& imga : pattern_gamma) {
		cv::Mat img;
		cv::normalize(imga, img, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("pattern", img);
		cv::waitKey(0);
		cv::destroyWindow("pattern");
	}*/


	std::vector<cv::Mat> pattern_luminance;
	if (luminance) {
		for (const auto& img : pattern_gamma) {
			pattern_luminance.emplace_back(
				m_img_processing->simulate_luminance(
					img,
					100,
					0.2745,
					1920,
					1080,
					0.0,
					0.0,
					0.0,
					0.0)
			);
		}
	}
	else pattern_luminance = pattern_gamma;

	for (const auto& imga : pattern_luminance) {
		cv::Mat img;
		cv::normalize(imga, img, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::imshow("pattern", img);
		cv::waitKey(0);
		cv::destroyWindow("pattern");
	}

	std::vector<cv::Mat> pattern_realistic;

	
	for (const auto& img : pattern_gamma) {
		pattern_realistic.emplace_back(createRealisticFromPattern(
			img,
			aperture_number,
			dispaly_pixel_pitch,
			distance,
			object_height,
			image_height,
			dest_width,      
			dest_height,    
			homography_points));
	}

	for (const auto& imga : pattern_realistic) {
		cv::Mat img;
		cv::normalize(imga, img, 0, 255, cv::NORM_MINMAX, CV_8U);
		cv::namedWindow("Pattern", cv::WINDOW_NORMAL);

		cv::resizeWindow("Pattern", 1500, 1000);
		cv::imshow("Pattern", img);
		cv::waitKey(0);
		cv::destroyWindow("Pattern");
	}

}

// If no smoothing should be done set Image height to zero!
// // If no homogrpahy should be done. set destWidth or destheight to zero!
cv::Mat Deflectometry::createRealisticFromPattern(
	const cv::Mat& pattern,
	const double aperture_number,
	const double dispaly_pixel_pitch,
	const double distance,   
	const double object_height, 
	const double image_height, 
	const int dest_width , 
	const int dest_height,   
	const RoiBorders<double> destination)
{
	CV_Assert(aperture_number >= 0 && image_height >= 0);
	CV_Assert(distance >= 0 && object_height > 0);
	CV_Assert(dest_width >= 0 && dest_height >= 0);

	const double scale = object_height/image_height;
	const double focal_length = distance * scale / (scale + 1); // g = (m+1)/m * f
	const double circ_entrance_pupil = focal_length / aperture_number; // f# = f/D -> D = entrance pupil 
	cv::Mat smoothed, warped;
	cv::Size output_size{ dest_width, dest_height };

	// Smoothing 
	if (image_height) {
		const int n_pixel_diameter =
			static_cast<int>(std::ceil(circ_entrance_pupil / dispaly_pixel_pitch));
		cv::Mat circular_binary = m_img_processing->createCirculeBinaryMask(n_pixel_diameter / 2);
		double sum = cv::sum(circular_binary)[0];
		circular_binary /= sum;
		cv::filter2D(pattern, smoothed, CV_64F, circular_binary);
	}
	else smoothed = pattern;
	
	// Warp the image 
	if (dest_height && dest_width) {
		RoiBorders<int> source{
			{0,0},
			{pattern.cols - 1, 0},
			{0, pattern.rows - 1},
			{pattern.cols - 1, pattern.rows - 1}

		};

		const cv::Mat homography =
			m_img_processing->getHomographyMat(source, destination);

		cv::warpPerspective(pattern, warped, homography, output_size);
	}
	else warped = smoothed;

	return warped;
}

std::vector<cv::Mat> Deflectometry::generatePattern(
	const Shift_mode mode,
	const CalibrationMethod method,
	const std::string& calib_path,
	const FrameRole dst,
	bool save_frames,
	const std::string& path)
{
	if (save_frames) CV_Assert(!path.empty());

	setupPattern(*m_img_store);
	
	setupCalibration(method, calib_path);

	std::vector<cv::Mat> pattern_double = 
		m_pattern->generate_phaseShift(
			mode, 
			127.5,
			127.5,
			10,
			1920,
			1080,
			UniformRowsCols{});

	//std::vector<cv::Mat> pattern_double = m_img_store->get(FrameRole::PatternDouble);

	if (save_frames) {
		m_img_store->saveRoleXML(FrameRole::PatternDouble, path);
		m_img_store->saveRoleXML(FrameRole::RawPhase, path);
	}
	
	m_img_store->show(FrameRole::PatternDouble);
	return pattern_double;
}



std::vector<cv::Mat> Deflectometry::do_phase_measurement(
	Shift_mode mode,
	int n_pics_per,
	bool save,
	const std::string& save_path,
	int n_periods)
{
	// Fixed Destination of the Gray Calibration File
	setupPattern(*m_img_store);

	setupCalibration(CalibrationMethod::None, "");
	
	std::vector<cv::Mat> pattern = 
		m_pattern->generate_phaseShift(
			mode, 
			127.5,
			127.5,
			n_periods,
			1920,
			1080,
			UniformRowsCols{});



	if (!init()) {
		std::cerr << "Init Method failed. Stop Meassurment \n"; 
		return {};
	}

	if (!m_pattern_config.empty()) throw std::runtime_error("m_pattern_config does already contain data \n");

	for (const auto& cfg : m_pattern->getPhaseConfig()) {
		m_pattern_config.emplace_back(std::make_unique<defl::PhaseShiftConfig>(cfg));
	}
	
	logging(save_path);

	int expected_patterns{ 0 };
	for (const auto& m : m_pattern_config) {
		expected_patterns += (m->steps) * 2; // steps*2 for horizontal and vertical
	}

	CV_Assert(expected_patterns == m_img_store->count(FrameRole::Pattern) &&
		"The ammount of patterns differs to expected value");

	std::vector<cv::Mat> patterns = m_img_store->get(FrameRole::Pattern);
	std::vector<cv::Mat>::const_iterator iter = patterns.cbegin();
	std::vector<cv::Mat>::const_iterator end_iter = patterns.cend();
	
	// Assign category for save class
	m_acquisition_controller->mode = FrameRole::RawInput;

	// Start Acquisitionworker and ScreenDisplay
	m_screenDisplay->assignCameraPreProcessing(showRawMaxValred);
	m_screenDisplay->start();
	m_acquisition_worker->start();
	
	std::cout << "Press ENTER to start gray-value calibration...\n";
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	std::unique_lock img_save_lock(m_acquisition_controller->mtx);

	int n_frequencies = static_cast<int>(m_pattern_config.size());
	for (int i = 0; i < n_frequencies; ++i) {
		int n_shifts = m_pattern_config.at(i)->steps;
		for (int y = 0; y < n_shifts * 2; ++y) { //n_shifts*2 for horizontral and vertical
			if (iter == end_iter) {throw std::runtime_error("Iterator is not in boundaries \n"); }
			m_screenDisplay->showPattern(*iter);
			for (int x = 0; x < n_pics_per; ++x) {
				if(!x) std::this_thread::sleep_for(std::chrono::milliseconds(1100));
				m_acquisition_worker->requestSave();
				m_acquisition_controller->cv.wait(img_save_lock);
				std::this_thread::sleep_for(std::chrono::milliseconds(600));
			}
			iter++;
		}
	}
	
	img_save_lock.unlock();

	m_acquisition_worker->stop();
	m_screenDisplay->stop();
	if (save) {
		//m_img_store->saveRoleXML(FrameRole::PatternDouble, save_path);
		m_img_store->saveRole(FrameRole::RawInput, save_path);
		m_img_store->saveRoleXML(FrameRole::RawInput, save_path);
		m_img_store->saveRole(FrameRole::Pattern, save_path);
		m_img_store->saveRoleXML(FrameRole::Pattern, save_path);
		m_img_store->saveRole(FrameRole::RawPhase, save_path);
		m_img_store->saveRoleXML(FrameRole::RawPhase, save_path);
	}
	if (!disconnect()) {
		std::cerr << "The Class AcquisitionWorker and Class could not be destroyed \n";
	}

	return m_img_store->get(FrameRole::RawInput);
}

void Deflectometry::load(FrameRole role, const std::string& path) {
	m_img_store->loadRoleXML(role, path);
}

std::vector<cv::Mat> Deflectometry::get(FrameRole role) {
	return m_img_store->get(role);
}

std::vector<cv::Mat> Deflectometry::do_wrapped_phase(
	const std::vector<cv::Mat>& vec,
	int n_pics_perPhase,
	int n_shifts,
	bool save,
	const std::string& path_save) 
{
	std::vector<cv::Mat> wrapped_phase_out = 
		m_img_processing->do_wrapped_Phase(vec, n_pics_perPhase, n_shifts);

	// --- Saving of the images ---
	for (const auto& m : std::initializer_list<std::size_t>{ 0,1 }) {
		m_img_store->add(FrameRole::WrappedPhase, wrapped_phase_out[m]);
	}
	for (const auto& m : std::initializer_list<std::size_t>{ 2,3 }) {
		m_img_store->add(FrameRole::Contrast, wrapped_phase_out[m]);
	}
	for (const auto& m : std::initializer_list<std::size_t>{ 4,5 }) {
		m_img_store->add(FrameRole::BaseIntensity, wrapped_phase_out[m]);
	}

	if (save) {
		m_img_store->saveRole(FrameRole::WrappedPhase, path_save);
		m_img_store->saveRoleXML(FrameRole::WrappedPhase, path_save);
		m_img_store->saveRole(FrameRole::Contrast, path_save);
		m_img_store->saveRoleXML(FrameRole::Contrast, path_save);
		m_img_store->saveRole(FrameRole::BaseIntensity, path_save);
		m_img_store->saveRoleXML(FrameRole::BaseIntensity, path_save);
	}
	std::cout << "Show Raw Phase \n";
	m_img_store->show(FrameRole::RawPhase);
	std::cout << "Show wrapped PHase \n";
	m_img_store->show(FrameRole::WrappedPhase);
	m_img_store->show(FrameRole::Contrast);
	m_img_store->show(FrameRole::BaseIntensity);
	return std::vector<cv::Mat>(wrapped_phase_out.begin(), std::next(wrapped_phase_out.begin(), 2));
}



bool Deflectometry::setupCalibration(
	const CalibrationMethod method,
	const std::string& path)
{
	CV_Assert(m_pattern != nullptr);
	CV_Assert(m_img_processing != nullptr);

	if (m_calibration == nullptr) {
		m_calibration = std::make_unique<GrayCalibration>(*m_img_store);
	}

	switch (method) {
	case(CalibrationMethod::None):
		std::cout << "No calibration Mode selected !\n";
		break;
	case(CalibrationMethod::Lut): {
		std::cout << "Calibratio via Lut selected \n" << "Try loading LUT... \n";
		CV_Assert(!path.empty());
		m_img_store->loadRoleXML(FrameRole::GrayLUT, path);
		m_calibration->setupCalibrationMethod(
			gr_calib::LUTCalibration{}, m_img_store->getLut());
		std::cout << "LUT loaded and ready \n";
		break;
	}
	case(CalibrationMethod::Active): {
		std::cout << "Active Calibration selected \n" << "Try loading images ... \n";
		CV_Assert(!path.empty());
		m_img_store->loadRoleXML(FrameRole::AcitveGrayCalib, path);
		std::vector<cv::Mat> active_gray = m_img_store->get(FrameRole::AcitveGrayCalib);
		CV_Assert(!active_gray.empty());

		std::cout << "This path is not implemented at this point ";
		throw std::runtime_error("This path must be terminated \n");
		break;
	}
	case(CalibrationMethod::Passive):
		std::cout << "Passive Calibration iselected \n" << "Try loading images ... \n";
		CV_Assert(!path.empty());
		m_img_store->loadRoleXML(FrameRole::PassiveGrayCalib, path);
		std::vector<cv::Mat> passive_gray = m_img_store->get(FrameRole::PassiveGrayCalib);
		CV_Assert(!passive_gray.empty());

		std::cout << "This path is not implemented at this point ! \n";
		throw std::runtime_error("Not implemente ");
		break;
	}
	return true;
}


void Deflectometry::setupPattern(
	ImageStore& img_store,
	int pixelX,
	int pixelY) 
{
	//CV_Assert(!path.empty());
	CV_Assert(pixelX >= 1);
	CV_Assert(pixelY >= 1);

	if (m_pattern != nullptr) {
		return;
	}
	m_pattern = std::make_unique<Pattern>(img_store);
}

std::vector<cv::Mat> Deflectometry::acquire_img(
	const std::vector<cv::Mat>& src,
	const FrameRole dst,
	const int n_pics_per_value)
{
	CV_Assert(n_pics_per_value >= 1);

	if (!init()) {
		std::cerr << "Init Method failed. Stop GrayValueCalibration\n";
		return {};
	}

	auto iter = src.cbegin();
	auto iter_end = src.cend();

	m_acquisition_controller->mode = dst;              // FrameRole::GrayCalibrationCam;

	m_screenDisplay->assignCameraPreProcessing(showRawMaxValred);
	m_screenDisplay->start();
	m_acquisition_worker->start();

	std::cout << "Press ENTER to start gray-value calibration...\n";
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	// Lock für wait()
	std::unique_lock lk(m_acquisition_controller->mtx);

	// 3) Für jeden Grauwert
	for (size_t gray = 0; gray < src.size(); ++gray)
	{
		if (iter == iter_end)
			throw std::runtime_error("Pattern iterator out of bounds!");

		// Muster anzeigen
		m_screenDisplay->showPattern(*iter);

		for (int i = 0; i < n_pics_per_value; ++i)
		{
			if (i == 0)
				std::this_thread::sleep_for(std::chrono::milliseconds(500));

			m_acquisition_worker->requestSave();

			// Warte auf Save-Bestätigung aus AcquisitionWorker
			m_acquisition_controller->cv.wait(lk);
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		++iter;
	}

	// Stop
	lk.unlock();
	m_acquisition_worker->stop();
	m_screenDisplay->stop();
	disconnect();

	return m_img_store->get(dst);
}


std::vector<cv::Mat> Deflectometry::acquire_img(
	const FrameRole src,
	const FrameRole dst,
	const int n_pics_per_value) 
{
	CV_Assert(n_pics_per_value >= 1);

	if (!init()) {
		std::cerr << "Init Method failed. Stop GrayValueCalibration\n";
		return {};
	}

	// 2) Alle Grauwert-Bilder holen
	std::vector<cv::Mat> gray_patterns =
		m_img_store->get(src);

	return acquire_img(gray_patterns, dst, n_pics_per_value);
}

bool Deflectometry::logging(
	const std::string& path)
{
	CV_Assert(!path.empty());
	CV_Assert(std::filesystem::exists(std::filesystem::path(path)));
	bool sucess{ true };

	// Camera Config
	if (m_camera_config.empty()) {
		std::cout << "No camera config available \n";
		sucess = false;
	}
	else {
		int counter{};
		for (const auto& cfg : m_camera_config) {
			std::string camera_config{ path + "/" + "CameraConfig" + std::to_string(counter++) + ".xml" };
			cfg->save_to_XML(camera_config);
		}
	}

	// Phase Config
	if (m_pattern_config.empty() && (m_pattern != nullptr)) {
		for (const auto& cfg : m_pattern->getPhaseConfig()) {
			m_pattern_config.emplace_back(std::make_unique<defl::PhaseShiftConfig>(cfg));
		}
	}
	{
		int counter{};
		for (const auto& cfg : m_pattern_config) {
			std::string camera_config{ path + "/" + "PatternConfig" + std::to_string(counter++) + ".xml" };
			cfg->save_to_XML(camera_config);
		}
	}
	// Gray Calibration Config



	return sucess;
}

GrayCalibVector Deflectometry::calc_response_curve_sections(
	const int n_pics_per_value,
	const int n_steps,
	bool save,
	const std::string& save_path,
	const int sections_x,
	const int sections_y)
{
	CV_Assert(n_pics_per_value >= 1);
	CV_Assert(n_steps >= 1);
	if (save) CV_Assert(!save_path.empty());
	CV_Assert(sections_x >= 1);
	CV_Assert(sections_y >= 1);

	// Creates patter instance
	setupPattern(*m_img_store);

	setupCalibration(CalibrationMethod::None, "");

	if (init()) {
		std::cout << "Hardware connection failed. \nReturn to Caller\n";
		return {};
	}

	// Creates the pattern
	std::vector<cv::Mat> gray_calibGT =
		m_pattern->generateGrayCalibrationSequence(n_steps);

	std::vector<cv::Mat> gray_calib = acquire_img(
		gray_calibGT,
		FrameRole::GrayCalibrationCam,
		n_pics_per_value);

	CV_Assert(!gray_calib.empty());

	GrayCalibVector borders = borderPoints_gray_val(1920, 1080, sections_x, sections_y);
	
	m_img_processing->getResponseCurve_perSection(
		gray_calib,
		borders,
		n_pics_per_value,
		n_steps);

	if (save) {
		borders.save(save_path);
	}

	return borders;
}

bool Deflectometry::do_grayvalue_calibration(
	const int n_pics_per_value,
	const int n_steps,
	bool save,
	const std::string& path)
{
	CV_Assert(n_pics_per_value >= 1);
	CV_Assert(n_steps >= 1);
	if (save) CV_Assert(!path.empty());
	
	// Creates patter instance
	setupPattern(*m_img_store);

	setupCalibration(CalibrationMethod::None, "");

	// Creates the pattern
	std::vector<cv::Mat> gray_calibGT =
		m_pattern->generateGrayCalibrationSequence(n_steps);

	std::vector<cv::Mat> gray_calib = acquire_img(
		gray_calibGT,
		FrameRole::GrayCalibrationCam,
		n_pics_per_value);

	CV_Assert(!gray_calib.empty());

	std::vector<std::pair<double,double>> LUT = 
		m_img_processing->gray_value_calib(gray_calib, n_pics_per_value, n_steps);

	m_img_store->add(FrameRole::GrayLUT, std::move(LUT));
	if(save)
		m_img_store->saveRole(FrameRole::GrayLUT, path);

	return true;
}

GrayCalibVector Deflectometry::borderPoints_gray_val(
	const int pixel_x,
	const int pixel_y,
	const int sections_x,
	const int sections_y)
{
	assert(pixel_x > 0 && pixel_y > 0);
	assert(sections_x >= 0 && sections_y >= 0);
	if (bool(sections_x) != bool(sections_y)) {
		std::cout << "Invalid Input in boderPoints_gray_val() \n" <<
			"Return to caller \n";
		return {};
	}
		
	int n_sections{}, rect_length_x{}, rect_length_y{};
	GrayCalibVector border_sections{};
	int area = pixel_x * pixel_y;
	
	if (sections_x && sections_y) {
		double pix_per_sec_x = static_cast<double>(pixel_x) /
			static_cast<double>(sections_x);
		rect_length_x = static_cast<int>(std::ceil(pix_per_sec_x));
		double pix_per_sec_y = static_cast<double>(pixel_y) /
			static_cast<double>(sections_y);
		rect_length_y = static_cast<int>(std::ceil(pix_per_sec_y));
		n_sections = sections_x * sections_y;
	}
	else {
		rect_length_x = rect_length_y = std::gcd(pixel_y, pixel_x);
		n_sections = area / (rect_length_x*rect_length_y);
	}
	
	border_sections.reserve(static_cast<std::size_t>(n_sections));
	
	std::pair<int, int> upper_left_corner{ 0,0 }; // x,y
	std::pair<int, int> shift_right{ rect_length_x, 0 };
	std::pair<int, int> shift_down{ 0, rect_length_y };

	for (std::size_t i = 0; i < static_cast<std::size_t>(n_sections); ++i) {
		std::pair<int, int> upper_right_corner = upper_left_corner + shift_right;
		if (upper_right_corner.first >= pixel_x) upper_right_corner.first = pixel_x;
		std::pair<int, int> down_left_corner = upper_left_corner + shift_down;
		if (down_left_corner.second >= pixel_y) down_left_corner.second = pixel_y;
		std::pair<int, int> down_right_corner{ upper_right_corner.first, down_left_corner.second };
		
		Gray_section_data data{};
		data.roi_img.left_up_corner = upper_left_corner;
		data.roi_img.right_up_corner = upper_right_corner;
		data.roi_img.left_down_corner = down_left_corner;
		data.roi_img.right_down_corner = down_right_corner;
		border_sections.push_back(data);
		if (upper_right_corner.first < pixel_x) upper_left_corner = upper_right_corner;
		else {
			upper_left_corner = std::pair<int, int>(0, down_right_corner.second);
		}
	}
	// Debug
	cv::Mat dummy = cv::Mat::zeros(pixel_y, pixel_x, CV_8U);
	for (const auto& border : border_sections) {
		cv::Point point{border.roi_img.left_up_corner.first, border.roi_img.left_up_corner.second };
		cv::drawMarker(dummy, point, cv::Scalar(255));
	}
	cv::imshow("Debug", dummy);
	cv::waitKey(0);
	cv::destroyWindow("Debug");

	return border_sections;
}

std::vector<std::pair<double,double>> Deflectometry::findParallelogramCorners(const cv::Mat& bin) {
	CV_Assert(!bin.empty());
	CV_Assert(bin.type() == CV_8U);
	return m_img_processing->find_ParallelogramCorners(bin);
}

cv::Mat Deflectometry::get_Homogrpahy_mat(
	const std::vector<std::pair<int, int>>& src,
	const std::vector<std::pair<double, double>>& dst,
	const int method
)
{
	CV_Assert(src.size() == dst.size());
	CV_Assert(!src.empty() && !dst.empty());

	std::vector<cv::Vec2d> src_n;
	std::vector<cv::Vec2d> dst_n;
	for (const auto& s_pts : src) {
		src_n.emplace_back(cv::Vec2d(s_pts.first, s_pts.second));
	}
	for (const auto& d_pts : dst) {
		dst_n.emplace_back(cv::Vec2d(d_pts.first, d_pts.second));
	}

	return get_Homogrpahy_mat(src_n, dst_n);
}

cv::Mat Deflectometry::get_Homogrpahy_mat(
	const std::vector<cv::Vec2d>& srcPts,
	const std::vector<cv::Vec2d>& dstPts,
	const int method)
{
	CV_Assert(srcPts.size() == dstPts.size());
	return m_img_processing->getHomographyMat(srcPts, dstPts, method);
}

cv::Vec2d Deflectometry::projectPoint_homography(
	const cv::Vec2d& img,
	const cv::Mat& homography) 
{
	CV_Assert(!homography.empty());
	return m_img_processing->projectPoint_homography(img, homography);
}

cv::Mat Deflectometry::subtractSurface(
	const cv::Mat& img,
	const cv::Mat& mask,
	bool show_subtracted)
{
	CV_Assert(img.channels() == 1 && mask.channels() ==1);
	CV_Assert(img.size() == mask.size());
	CV_Assert(img.type() == CV_64F);
	CV_Assert(mask.type() == CV_8U);
	
	cv::Mat out, norm;
	cv::Mat surface = 
		m_img_processing->fitSurface(img, mask);

	cv::subtract(img, surface, out, mask);

	if (show_subtracted) {
		cv::normalize(out, norm, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
		cv::imshow("Surface subtrakt", norm);
		cv::waitKey(0);
	}

	return surface;
}


std::vector<cv::Mat> Deflectometry::getFrames(
	FrameRole role,
	cv::Mat& img)
{
	m_screenDisplay->showPattern(img);
	return getFrames(role);
}

std::vector<cv::Mat> Deflectometry::getFrames(FrameRole role) {
	if (!init()) {
		std::cerr << "Init failed. Aborte ...\n";
		return {};
	}

	//Pics are saved in FrameRole::Calibration
	m_acquisition_controller->mode = role;

	m_screenDisplay->assignCameraPreProcessing(showRawMaxValred);
	m_screenDisplay->start();
	m_acquisition_worker->start();

	std::cout << "Camera calibration mode.\n"
		<< "Press S to save image\n"
		<< "Press E to exit\n\n";

	std::unique_lock lk(m_acquisition_controller->mtx);

	bool running = true;
	int imgCounter = 0;

	while (running)
	{
		char c = std::toupper(std::cin.get());
		if (!std::cin) {
			std::cin.clear();
			continue;
		}

		switch (c)
		{
		case 'S': {
			std::cout << "Saving image " << imgCounter << " ...\n";

			m_acquisition_worker->requestSave();

			// wait for worker to signal that the image is saved
			m_acquisition_controller->cv.wait(lk);

			++imgCounter;
			break;
		}

		case 'E':
			std::cout << "Exiting \n";
			running = false;
			break;

		default:
			std::cout << "Unknown key. Use S or E.\n";
			break;
		}
	}

	// stop worker/UI
	lk.unlock();
	m_acquisition_worker->stop();
	m_screenDisplay->stop();
	disconnect();

	std::vector<cv::Mat> checkerboard = m_img_store->get(role);
	return checkerboard;
}

// return RGB
std::array<double, (std::size_t)3> Deflectometry::doWhiteBalance(
	const std::vector<cv::Mat>& vec,
	int roi_x,
	int roi_y,
	int roi_width,
	int roi_height){
	CV_Assert(std::all_of(vec.begin(), vec.end(),
		[](const cv::Mat& mat) ->bool {
			return (!mat.empty() && (mat.type() == CV_8UC1));
		}));
	return m_img_processing->doWhiteBalance(vec, 300, 300, 300, 300);
}

bool Deflectometry::do_camera_calibration()
{
	std::string calibpath{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2026-01-08MAKOCalibration" };
	std::vector<cv::Mat> checkerboard = getFrames(FrameRole::Calibration);
	m_img_store->saveRole(FrameRole::Calibration, calibpath);

	// Save Path for Calibration in settings path
	std::array<std::vector<cv::Mat>, 2> camera = runCameraCalibration(checkerboard, true,
		std::string{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/in_VID5.xml" });

	for (const auto& m : camera[0]) {
		m_img_store->add(FrameRole::CalibImages, m);
	}

	m_img_store->saveRole(FrameRole::CalibImages, calibpath);
	return true;
}



// build function E(a,b)= sum 0...N-1 (y_i -(a*x_i+b^))^2
// optamisation challenge for a and b ...
// dE/da = sum 2(y_i+a*x_i-b)(-x_i)
// sum(x_i*y_i) - a*sum(x_i^2) -b*sum(x_i) = 0
// dE/db = sum2(y_i - a*x_i -b)(-1)
// sum(y_i) - a * sum(x_i) - b*N = 0
// = build LGS aA + b*B = C a*B+b*N = D
// A = sum(x_i)^2 B= sum(x_i) c = sum(x_i*y_i) D= sum (y_i)
std::pair<double, double> Deflectometry::fitLine1D(const std::vector<double>& y) {
	const int N = static_cast<int>(y.size());
	double sumx = 0.0, sumy = 0.0, sumxx = 0.0, sumxy = 0.0;

	for (int i = 0; i < N; ++i) {
		double x = static_cast<double>(i);
		double v = y[i];
		sumx += x;
		sumy += v;
		sumxx += x * x;
		sumxy += x * v;
	}


	double denom = N * sumxx - sumx * sumx;
	double a = (N * sumxy - sumx * sumy) / denom;
	double b = (sumy - a * sumx) / N;


	return { a, b };
}

void Deflectometry::saveSliceToCSV(const std::string& filename,
	const std::vector<double>& unwrap, 
	const std::pair<double, double>& unwrapFit,// regressions a,b unwrap
	const std::vector<double>& repro, 
	const std::pair<double, double>& reproFit)//regression a,b repro
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Could not open CSV file: " << filename << "\n";
		return;
	}

	file << "index,unwrap,unwrap_fit,unwrap_residual,repro,repro_fit,repro_residual\n";

	int N = std::min((int)unwrap.size(), (int)repro.size());
	for (int i = 0; i < N; ++i) {

		double unwrap_fit = unwrapFit.first * i + unwrapFit.second;
		double unwrap_residual = unwrap[i] - unwrap_fit;

		double repro_fit = reproFit.first * i + reproFit.second;
		double repro_residual = repro[i] - repro_fit;

		file << i << ","
			<< unwrap[i] << ","
			<< unwrap_fit << ","
			<< unwrap_residual << ","
			<< repro[i] << ","
			<< repro_fit << ","
			<< repro_residual << "\n";

	}

	file.close();
	std::cout << "CSV saved to " << filename << "\n";
}
