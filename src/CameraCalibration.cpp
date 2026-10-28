#include "CameraCalibration.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>

namespace {
cv::TermCriteria criteria(const CalibrationConfig& c) {
    return {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, c.maxIterations, c.epsilon};
}
cv::Mat checkedK(const cv::Mat& input, cv::Size size) {
    if (input.size() != cv::Size(3, 3) || input.channels() != 1)
        throw std::invalid_argument("Intrinsic guess must be a single-channel 3x3 matrix");
    cv::Mat k; input.convertTo(k, CV_64F); k = k.clone();
    if (!cv::checkRange(k) || k.at<double>(0, 0) <= 0 || k.at<double>(1, 1) <= 0 ||
        std::abs(k.at<double>(0, 1)) > 1e-12 || std::abs(k.at<double>(1, 0)) > 1e-12 ||
        std::abs(k.at<double>(2, 0)) > 1e-12 || std::abs(k.at<double>(2, 1)) > 1e-12 ||
        std::abs(k.at<double>(2, 2) - 1) > 1e-12 ||
        k.at<double>(0, 2) < 0 || k.at<double>(0, 2) >= size.width ||
        k.at<double>(1, 2) < 0 || k.at<double>(1, 2) >= size.height)
        throw std::invalid_argument("Invalid intrinsic guess (including principal point)");
    return k;
}
cv::Mat checkedD(const cv::Mat& input) {
    cv::Mat d = cv::Mat::zeros(5, 1, CV_64F);
    if (input.empty()) return d;
    if (input.channels() != 1 || (input.rows != 1 && input.cols != 1) ||
        (input.total() != 4 && input.total() != 5))
        throw std::invalid_argument("Only 4 or 5 Brown coefficients are supported");
    cv::Mat flat;
    input.clone().reshape(1, static_cast<int>(input.total())).convertTo(flat, CV_64F);
    if (!cv::checkRange(flat)) throw std::invalid_argument("Non-finite distortion guess");
    flat.copyTo(d.rowRange(0, flat.rows)); return d;
}
bool finiteResult(const MonoCalibrationResult& r) {
    if (r.cameraMatrix.empty() || r.distCoeffs.empty() ||
        !cv::checkRange(r.cameraMatrix) || !cv::checkRange(r.distCoeffs) ||
        r.cameraMatrix.at<double>(0, 0) <= 0 || r.cameraMatrix.at<double>(1, 1) <= 0 ||
        !std::isfinite(r.avgReprojectionError) || r.avgReprojectionError < 0) return false;
    for (const auto& v : r.rvecs) if (!cv::checkRange(v)) return false;
    for (const auto& v : r.tvecs) if (!cv::checkRange(v)) return false;
    return true; // Structural validity is NOT a certificate of parameter accuracy.
}
void writeConfig(cv::FileStorage& fs, const CalibrationConfig& c) {
    fs << "opencv_version" << CV_VERSION;
    fs << "board_width" << c.boardSize.width << "board_height" << c.boardSize.height;
    fs << "square_size" << c.squareSize << "model" << "Brown5";
    fs << "detector" << (c.useSBDetector ? "SB_NORMALIZE_ACCURACY" : "classic_cornerSubPix");
    fs << "subpixel_half_window" << c.subPixWindow;
    fs << "preprocessing_kernel_bias" << c.preprocessingKernelBias;
    fs << "max_iterations" << c.maxIterations << "epsilon" << c.epsilon;
    fs << "minimum_views" << c.minimumViews;
    fs << "initial_focal_length_px" << c.initialFocalLengthPx;
}
void writeMono(cv::FileStorage& fs, const CalibrationConfig& c, const MonoCalibrationResult& r) {
    fs << "image_width" << r.imageSize.width << "image_height" << r.imageSize.height;
    fs << "camera_matrix" << r.cameraMatrix << "distortion_coefficients" << r.distCoeffs;
    fs << "calibration_flags" << r.calibrationFlags;
    fs << "optimizer_rms" << r.optimizerRms << "avg_reprojection_error" << r.avgReprojectionError;
    fs << "accepted_image_indices" << r.acceptedImageIndices << "rejected_image_indices" << r.rejectedImageIndices;
    if (c.writePerViewErrors) fs << "per_view_reprojection_errors" << r.perViewErrors;
    if (c.writeExtrinsics && !r.rvecs.empty()) {
        cv::Mat extrinsics(static_cast<int>(r.rvecs.size()), 6, CV_64F);
        for (size_t i = 0; i < r.rvecs.size(); ++i) {
            cv::Mat row = extrinsics.row(static_cast<int>(i));
            r.rvecs[i].reshape(1, 1).copyTo(row.colRange(0, 3));
            r.tvecs[i].reshape(1, 1).copyTo(row.colRange(3, 6));
        }
        fs << "extrinsic_parameters" << extrinsics;
    }
}
} // namespace

bool CalibrationConfig::isValid() const {
    return boardSize.width > 2 && boardSize.height > 2 && std::isfinite(squareSize) && squareSize > 0 &&
        std::isfinite(aspectRatio) && aspectRatio > 0 && std::isfinite(principalPointGuess.x) &&
        std::isfinite(principalPointGuess.y) && minimumViews >= 3 && maxIterations > 0 &&
        std::isfinite(epsilon) && epsilon > 0 && subPixWindow > 0 &&
        preprocessingKernelBias >= -1 && preprocessingKernelBias <= 100 && !(fixK4 || fixK5 || fixK6) &&
        std::isfinite(initialFocalLengthPx) && initialFocalLengthPx > 0;
}
int CalibrationConfig::makeCalibrationFlags() const {
    int f = 0;
    if (useIntrinsicGuess) f |= cv::CALIB_USE_INTRINSIC_GUESS;
    if (fixPrincipalPoint) f |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if (zeroTangentDist) f |= cv::CALIB_ZERO_TANGENT_DIST;
    if (fixAspectRatio) f |= cv::CALIB_FIX_ASPECT_RATIO;
    if (fixK1) f |= cv::CALIB_FIX_K1;
    if (fixK2) f |= cv::CALIB_FIX_K2;
    if (fixK3) f |= cv::CALIB_FIX_K3;
    return f;
}
cv::Mat CameraCalibration::createInitialCameraMatrix(const cv::Size& size, const CalibrationConfig& c) {
    if (!c.isValid() || size.empty()) throw std::invalid_argument("Invalid initial camera configuration");
    cv::Mat k = cv::Mat::eye(3, 3, CV_64F);
    k.at<double>(0, 0) = c.initialFocalLengthPx;
    k.at<double>(1, 1) = c.initialFocalLengthPx / (c.fixAspectRatio ? c.aspectRatio : 1.0);
    k.at<double>(0, 2) = c.principalPointAtImageCenter ? (size.width - 1) * 0.5 : c.principalPointGuess.x;
    k.at<double>(1, 2) = c.principalPointAtImageCenter ? (size.height - 1) * 0.5 : c.principalPointGuess.y;

    return k;
}
cv::Mat CameraCalibration::toGray8(const cv::Mat& image) {
    if (image.empty()) throw std::invalid_argument("Empty input image");
    if (image.channels() != 1 && image.channels() != 3 && image.channels() != 4)
        throw std::invalid_argument("Only gray, BGR and BGRA images are supported");
    if (!cv::checkRange(image)) throw std::invalid_argument("Image contains non-finite values");
    cv::Mat work = image, gray;
    if (image.channels() > 1 && image.depth() != CV_8U && image.depth() != CV_16U && image.depth() != CV_32F)
        image.convertTo(work, CV_32F); // cvtColor does not support CV_64F
    if (work.channels() == 3) cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    else if (work.channels() == 4) cv::cvtColor(work, gray, cv::COLOR_BGRA2GRAY);
    else gray = work;
    if (gray.depth() != CV_8U) cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
    return gray;
}
bool CameraCalibration::detectChessboardCorners(const cv::Mat& image, const cv::Size& boardSize,
    std::vector<cv::Point2f>& corners, int halfWindow, bool useSB) {
    corners.clear();
    if (image.empty()) return false;
    if (boardSize.width <= 2 || boardSize.height <= 2 || halfWindow < 1)
        throw std::invalid_argument("Invalid board size or subpixel half-window");
    const cv::Mat gray = toGray8(image);
    bool found;
    if (useSB) {
        found = cv::findChessboardCornersSB(gray, boardSize, corners,
            cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_ACCURACY);
        // SB already returns subpixel points; do not apply cornerSubPix afterwards.
    } else {
        found = cv::findChessboardCorners(gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        if (found) cv::cornerSubPix(gray, corners, {halfWindow, halfWindow}, {-1, -1},
            {cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1e-4});
    }
    if (!found || corners.size() != static_cast<size_t>(boardSize.area())) { corners.clear(); return false; }
    return true;
}

std::vector<cv::Point3f> CameraCalibration::createBoardObjectPoints(const cv::Size& size, float square) {
    if (size.width < 1 || size.height < 1 || !std::isfinite(square) || square <= 0)
        throw std::invalid_argument("Invalid object-point grid");
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<size_t>(size.area()));
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) points.emplace_back(x * square, y * square, 0.0f);
    return points;
}
double CameraCalibration::computeReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& objects, const std::vector<std::vector<cv::Point2f>>& points,
    const std::vector<cv::Mat>& rv, const std::vector<cv::Mat>& tv,
    const cv::Mat& k, const cv::Mat& d, std::vector<float>& errors) 
{
    if (objects.empty() || objects.size() != points.size() || objects.size() != rv.size() || rv.size() != tv.size())
        throw std::invalid_argument("Reprojection input sizes differ or are empty");
    errors.clear(); double total = 0; size_t count = 0;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].empty() || objects[i].size() != points[i].size())
            throw std::invalid_argument("Per-view point counts differ or are empty");
        std::vector<cv::Point3d> object64;
        for (const auto& p : objects[i]) object64.emplace_back(p.x, p.y, p.z);
        std::vector<cv::Point2d> projected;
        cv::projectPoints(object64, rv[i], tv[i], k, d, projected);
        double err2 = 0;
        for (size_t j = 0; j < projected.size(); ++j) {
            const double dx = points[i][j].x - projected[j].x, dy = points[i][j].y - projected[j].y;
            err2 += dx * dx + dy * dy;
        }
        errors.push_back(static_cast<float>(std::sqrt(err2 / projected.size())));
        total += err2; count += projected.size();
    }
    return std::sqrt(total / count);
}
MonoCalibrationResult CameraCalibration::calibrateMonoSyn(
    const std::vector<std::vector<cv::Point2f>>& points, const CalibrationConfig& c,
    const cv::Mat& guessK, const cv::Mat& guessD, const cv::Size& size, bool drawDetectedCorners) {
    MonoCalibrationResult r;
    (void)drawDetectedCorners; // This point-only API has no images to draw on.
    try {
        if (!c.isValid() || size.empty()) throw std::invalid_argument("Invalid calibration configuration or image size");
        if (points.size() < static_cast<size_t>(c.minimumViews)) throw std::invalid_argument("Too few detected views");
        for (const auto& view : points) {
            if (view.size() != static_cast<size_t>(c.boardSize.area())) throw std::invalid_argument("Wrong corner count");
            for (const auto& p : view)
                if (!std::isfinite(p.x) || !std::isfinite(p.y)) throw std::invalid_argument("Non-finite image point");
        }
        if (!c.useIntrinsicGuess && !c.principalPointAtImageCenter)
            throw std::invalid_argument("A noncentral principal-point guess requires useIntrinsicGuess");
        r.imageSize = size;
        r.cameraMatrix = c.useIntrinsicGuess ? checkedK(guessK, size) : createInitialCameraMatrix(size, c);
        r.distCoeffs = c.useIntrinsicGuess ? checkedD(guessD) : cv::Mat::zeros(5, 1, CV_64F);
        if (c.fixAspectRatio && c.useIntrinsicGuess &&
            std::abs(r.cameraMatrix.at<double>(0, 0) / r.cameraMatrix.at<double>(1, 1) - c.aspectRatio) > 1e-6)
            throw std::invalid_argument("aspectRatio conflicts with intrinsic guess");
        r.calibrationFlags = c.makeCalibrationFlags();
        const std::vector<std::vector<cv::Point3f>> objects(points.size(), createBoardObjectPoints(c.boardSize, c.squareSize));
        r.optimizerRms = cv::calibrateCamera(objects, points, size, r.cameraMatrix, r.distCoeffs,
            r.rvecs, r.tvecs, r.calibrationFlags, criteria(c));
        r.avgReprojectionError = computeReprojectionErrors(objects, points, r.rvecs, r.tvecs,
            r.cameraMatrix, r.distCoeffs, r.perViewErrors);
        r.acceptedImageIndices.resize(points.size());
        std::iota(r.acceptedImageIndices.begin(), r.acceptedImageIndices.end(), 0);
        r.success = finiteResult(r);
        if (!r.success) r.errorMessage = "Calibration returned invalid parameters";
    } catch (const std::exception& e) { r.success = false; r.errorMessage = e.what(); }
    if (!r.success) std::cerr << "Calibration failed: " << r.errorMessage << '\n';
    return r;
}
MonoCalibrationResult CameraCalibration::calibrateMono(const std::vector<cv::Mat>& images,
    const CalibrationConfig& config, bool draw, int halfWindow) const {
    auto c = config;
    if (halfWindow == -1) halfWindow = c.subPixWindow;
    c.subPixWindow = halfWindow;
    std::vector<std::vector<cv::Point2f>> points;
    std::vector<int> accepted, rejected; cv::Size size;
    try {
        if (!c.isValid()) throw std::invalid_argument("Invalid configuration");
        for (size_t i = 0; i < images.size(); ++i) {
            if (images[i].empty()) { rejected.push_back(static_cast<int>(i)); continue; }
            if (size.empty()) size = images[i].size();
            if (images[i].size() != size) throw std::invalid_argument("Mixed image resolutions");
            std::vector<cv::Point2f> corners;
            if (!detectChessboardCorners(images[i], c.boardSize, corners, halfWindow, c.useSBDetector)) {
                rejected.push_back(static_cast<int>(i)); continue;
            }
            accepted.push_back(static_cast<int>(i)); points.push_back(corners);
            if (draw) {
                cv::Mat vis = toGray8(images[i]).clone();
                cv::drawChessboardCorners(vis, c.boardSize, corners, true);
                cv::imshow("Mono calibration corners", vis); cv::waitKey(50);
            }
        }
        const cv::Mat initial = size.empty() ? cv::Mat{} : createInitialCameraMatrix(size, c);
        auto r = calibrateMonoSyn(points, c, initial, cv::Mat{}, size);
        r.acceptedImageIndices = accepted; r.rejectedImageIndices = rejected;
        if (r.success && !c.outputFileName.empty() && !saveMonoCalibration(c.outputFileName, c, r))
            std::cerr << "Warning: calibration could not be saved\n";
        return r;
    } catch (const std::exception& e) { MonoCalibrationResult r; r.errorMessage = e.what(); return r; }
}
StereoCalibrationResult CameraCalibration::calibrateStereo(const std::vector<StereoImagePair>& pairs,
    const CalibrationConfig& config, bool draw, int halfWindow) const {
    StereoCalibrationResult result; auto c = config;
    if (halfWindow == -1) halfWindow = c.subPixWindow;
    c.subPixWindow = halfWindow;
    try {
        if (!c.isValid()) throw std::invalid_argument("Invalid stereo configuration");
        std::vector<std::vector<cv::Point2f>> left, right;
        std::vector<int> accepted, rejected; cv::Size size;
        for (size_t i = 0; i < pairs.size(); ++i) {
            const auto& p = pairs[i];
            if (p.left.empty() || p.right.empty()) { rejected.push_back(static_cast<int>(i)); continue; }
            if (size.empty()) size = p.left.size();
            if (p.left.size() != size || p.right.size() != size) throw std::invalid_argument("Mixed stereo resolutions");
            std::vector<cv::Point2f> a, b;
            if (!detectChessboardCorners(p.left, c.boardSize, a, halfWindow, c.useSBDetector) ||
                !detectChessboardCorners(p.right, c.boardSize, b, halfWindow, c.useSBDetector)) {
                rejected.push_back(static_cast<int>(i)); continue;
            }
            left.push_back(a); right.push_back(b); accepted.push_back(static_cast<int>(i));
            if (draw) {
                cv::Mat l = toGray8(p.left).clone(), r = toGray8(p.right).clone();
                cv::drawChessboardCorners(l, c.boardSize, a, true); cv::drawChessboardCorners(r, c.boardSize, b, true);
                cv::imshow("Stereo left corners", l); cv::imshow("Stereo right corners", r); cv::waitKey(50);
            }
        }
        if (size.empty()) throw std::invalid_argument("No stereo images");
        const cv::Mat initial = createInitialCameraMatrix(size, c);
        result.left = calibrateMonoSyn(left, c, initial, cv::Mat{}, size);
        result.right = calibrateMonoSyn(right, c, initial, cv::Mat{}, size);
        if (!result.left.success || !result.right.success) return result;
        const std::vector<std::vector<cv::Point3f>> objects(left.size(), createBoardObjectPoints(c.boardSize, c.squareSize));
        const int flags = c.fixIntrinsicsInStereo ? cv::CALIB_FIX_INTRINSIC :
            (c.makeCalibrationFlags() | cv::CALIB_USE_INTRINSIC_GUESS);
        result.stereoError = cv::stereoCalibrate(objects, left, right, result.left.cameraMatrix, result.left.distCoeffs,
            result.right.cameraMatrix, result.right.distCoeffs, size, result.R, result.T, result.E, result.F, flags, criteria(c));
        // Refit per-camera poses after K/D changed. These are MONO diagnostics,
        // not the joint stereo residuals reported in stereoError.
        for (size_t side = 0; side < 2; ++side) {
            auto& r = side == 0 ? result.left : result.right;
            const auto& points = side == 0 ? left : right;
            for (size_t i = 0; i < points.size(); ++i)
                if (!cv::solvePnP(objects[i], points[i], r.cameraMatrix, r.distCoeffs,
                    r.rvecs[i], r.tvecs[i], true, cv::SOLVEPNP_ITERATIVE)) throw std::runtime_error("Pose refit failed");
            r.avgReprojectionError = computeReprojectionErrors(objects, points, r.rvecs, r.tvecs,
                r.cameraMatrix, r.distCoeffs, r.perViewErrors);
            r.optimizerRms = r.avgReprojectionError; r.acceptedImageIndices = accepted; r.rejectedImageIndices = rejected;
            r.success = finiteResult(r);
        }
        result.success = result.left.success && result.right.success && std::isfinite(result.stereoError) &&
            cv::checkRange(result.R) && cv::checkRange(result.T) && cv::checkRange(result.E) && cv::checkRange(result.F);
        if (result.success && !c.outputFileName.empty() && !saveStereoCalibration(c.outputFileName, c, result))
            std::cerr << "Warning: stereo result could not be saved\n";
    } catch (const std::exception& e) { result.success = false; std::cerr << "Stereo failed: " << e.what() << '\n'; }
    return result;
}
bool CameraCalibration::saveMonoCalibration(const std::string& name, const CalibrationConfig& c,
    const MonoCalibrationResult& r) const {
    if (!r.success || name.empty()) return false;
    try {
        cv::FileStorage fs(name, cv::FileStorage::WRITE); if (!fs.isOpened()) return false;
        writeConfig(fs, c); writeMono(fs, c, r); return true;
    } catch (const cv::Exception& e) { std::cerr << e.what() << '\n'; return false; }
}
bool CameraCalibration::saveStereoCalibration(const std::string& name, const CalibrationConfig& c,
    const StereoCalibrationResult& r) const {
    if (!r.success || name.empty()) return false;
    try {
        cv::FileStorage fs(name, cv::FileStorage::WRITE); if (!fs.isOpened()) return false;
        writeConfig(fs, c);
        fs << "image_width" << r.left.imageSize.width << "image_height" << r.left.imageSize.height;
        fs << "fix_intrinsics_in_stereo" << static_cast<int>(c.fixIntrinsicsInStereo);
        fs << "stereo_flags" << (c.fixIntrinsicsInStereo ? cv::CALIB_FIX_INTRINSIC :
            (c.makeCalibrationFlags() | cv::CALIB_USE_INTRINSIC_GUESS));
        fs << "camera_matrix1" << r.left.cameraMatrix << "distortion_coefficients1" << r.left.distCoeffs;
        fs << "camera_matrix2" << r.right.cameraMatrix << "distortion_coefficients2" << r.right.distCoeffs;
        fs << "avg_reprojection_error_cam1" << r.left.avgReprojectionError;
        fs << "avg_reprojection_error_cam2" << r.right.avgReprojectionError;
        fs << "left" << "{"; writeMono(fs, c, r.left); fs << "}";
        fs << "right" << "{"; writeMono(fs, c, r.right); fs << "}";
        fs << "R" << r.R << "T" << r.T << "E" << r.E << "F" << r.F << "stereo_error" << r.stereoError;
        return true;
    } catch (const cv::Exception& e) { std::cerr << e.what() << '\n'; return false; }
}

// New Line is outer std::vector, while all points on a line are listed in the inner std::vector
DistortionCalibrationResult CameraCalibration::DistortionCalibration(
    const std::vector<std::vector<cv::Point2d>>& imagePts,
    const DistortionCalibrationConfig& config) 
{
    
    // --- Radial partial derivatives ---
    auto XAfterB = [](const double& x_i, const double& r2) -> double {
        return x_i * r2;
        };
    auto XAfterC = [](const double& x_i, const double& r4) -> double {
        return x_i * r4;
        };
    auto XAfterD = [](const double& x_i, const double& r6) -> double {
        return x_i * r6;
        };
    constexpr double (*YAfterB)(const double& y_i, const double& r2) = XAfterB;
    constexpr double (*YAfterC)(const double& y_i, const double& r4) = XAfterC;
    constexpr double (*YAfterD)(const double& y_i, const double& r6) = XAfterD;

    // --- Tangential partial derivatives ---
    auto XAfterP1 = [](const double& x_i, const double& y_i) -> double {
        return 2.0 * x_i * y_i;
        };
    auto XAfterP2 = [](const double& r2, const double& x_i2) -> double {
        return r2 + 2 * x_i2;
        };
    constexpr double (*YAfterP1)(const double& r2, const double& y_i2) = XAfterP2; 
    constexpr double (*YAfterP2)(const double& x_i, const double& y_i) = XAfterP1;

    // --- Radius derivatives --- 
    auto RAfterX_i_PBS = [](const double& x2, const double& y2, const double& x) -> double {
        const double r = std::sqrt(x2 + y2);
        // At the origin, zero is a placeholder for use in the
        // polynomial chain-rule expressions below.
        return r > 0.0 ? x / r : 0.0;
        };
    constexpr double (*RAfterY_i_PBS)(const double& x2, const double& y2, const double& y) = RAfterX_i_PBS;
    constexpr double (*RAfterX_i_mess)(const double& x2, const double& y2, const double& x) = RAfterX_i_PBS;
    constexpr double (*RAfterY_i_mess)(const double& x2, const double& y2, const double& y) = RAfterX_i_PBS;
    
    // --- Derivatives with respect to measured coordinates ---
    auto XAfterX_something = [](
        const double& x_i, const double& y_i,
        const double& b, const double& c, const double& d,
        const double& p1, const double& p2,
        const double& r2, const double& r4) -> double
        {
            return 1.0 + b * r2 + c * r4 + d * r2 * r4
                + 2.0 * x_i * x_i * (b + 2.0 * c * r2 + 3.0 * d * r4)
                + 2.0 * p1 * y_i
                + 6.0 * p2 * x_i;
        };
    auto XAfterY_something = [](
        const double& x_i, const double& y_i,
        const double& b, const double& c, const double& d,
        const double& p1, const double& p2,
        const double& r2, const double& r4) -> double
        {
            return 2.0 * x_i * y_i * (b + 2.0 * c * r2 + 3.0 * d * r4)
                + 2.0 * p1 * x_i
                + 2.0 * p2 * y_i;
        };
    constexpr double (*YAfterX_something)(
        const double& x_i, const double& y_i,
        const double& b, const double& c, const double& d,
        const double& p1, const double& p2,
        const double& r2, const double& r4
        ) = XAfterY_something;

    // Call with coordinates AND tangential coefficients swapped.
    constexpr double (*YAfterY_something)(
        const double& y_i, const double& x_i,
        const double& b, const double& c, const double& d,
        const double& p2, const double& p1,
        const double& r2, const double& r4
        ) = XAfterX_something;
    
    constexpr std::tuple func_ptr{
        +XAfterB, // 0
        +XAfterC, // 1
        +XAfterD, // 2
        +XAfterP1, // 3
        +XAfterP2, // 4
        +XAfterX_something, // 5
        +RAfterX_i_PBS, // 6
        RAfterX_i_mess, // 7
        YAfterB, // 8
        YAfterC, // 9
        YAfterD, // 10
        YAfterP1, // 11
        YAfterP2, // 12
        YAfterY_something, // 13
        RAfterY_i_PBS, // 14
        RAfterY_i_mess, // 15
        +XAfterY_something, // 16
        YAfterX_something // 17
    };

    auto build_Error_pointRefinement = [](
        const cv::Mat& a, // Derivative after Distortion coefficients
        const cv::Mat& parameters, // parameters of the coefficients 
        const cv::Mat& b, // Derivative after the meassuring points
        const cv::Mat& point_Corrections,
        cv::Mat& residual
        ) -> void {
            residual = a * parameters + b * point_Corrections;
        };

    auto build_Error = [](
        const cv::Mat& a,
        const cv::Mat& x,
        cv::Mat& residual
        ) -> void
        {
            residual = a * x;
        };

    // Lambda for creating all Jacobi values for one given row
    auto build_Row = [&func_ptr](
        const std::size_t& radial_coefficients, // [0 ... 3] possible values
        const std::size_t& tangential_coefficients, // true false, if true 2 parameters are calculated 
        const std::array<const cv::Point2d*, 3>& points, // the three point over which we try to build the "Row"
        const std::array<const cv::Point2d*, 3>& point_correction,
        const cv::Mat& parameters,
        cv::Mat jacobi_row_out, // Output row for the jacobi
        double& residual,
        const bool point_refinement) -> void
        {
            assert(jacobi_row_out.cols == 1);
            assert(jacobi_row_out.type() == CV_64F);

            assert(radial_coefficients <= 3);
            assert(tangential_coefficients == 0 ||
                tangential_coefficients == 2);

            assert(parameters.type() == CV_64F);
            assert(parameters.cols == 1);
            assert(parameters.rows ==
                static_cast<int>(
                    radial_coefficients + tangential_coefficients + 2));

            //std::array<cv::Point2d, 3> corrected_ptss{};

            // in This container the Corrected Pts are stored for calculating the residual
            // key: 0 -> X1, key: 1 -> X2, key: 2 -> X3, key: 3 -> Y1 ...  
            std::map<int, double> corrected_pts{};

            // If tangential Parameters is on we have P1 and P2 additional plus the Point of Best Symmetrie in every case 
            const std::size_t n_paramters =
                radial_coefficients + tangential_coefficients + 2
                + (point_refinement ? 6 : 0);

            assert(jacobi_row_out.rows == static_cast<int>(n_paramters) &&
                "Parameter vector must fit the specified ammount of values in size \n");

            // The first to need to be subtracted the rest is positive
            constexpr std::array<std::pair<int, int>, 6> indices{ {
                {-1,0}, {0,-2}, {-2,1}, {2,0}, {0,1}, {1,2}
            } };
            // only used if point refinement is off 
            const cv::Point2d dummy{ 0.0, 0.0 };

            enum Parameters {
                k1,
                k2,
                k3,
                P1,
                P2,
                X_PBS,
                Y_PBS,
                DeltaX1,
                DeltaY1,
                DeltaX2,
                DeltaY2,
                DeltaX3,
                DeltaY3
            };

            auto getCurrentParameter = [](
                const std::size_t& radial_coefficients,
                const std::size_t& tangential_coefficients,
                const bool& point_refinement,
                const std::size_t& index) -> Parameters
                {
                    if (index < radial_coefficients)
                        return static_cast<Parameters>(index);

                    const std::size_t tangentialEnd =
                        radial_coefficients + tangential_coefficients;

                    if (index < tangentialEnd)
                        return static_cast<Parameters>(
                            static_cast<std::size_t>(Parameters::P1)
                            + index - radial_coefficients);

                    const std::size_t pbsEnd = tangentialEnd + 2;

                    if (index < pbsEnd)
                        return static_cast<Parameters>(
                            static_cast<std::size_t>(Parameters::X_PBS)
                            + index - tangentialEnd);

                    if (point_refinement && index < pbsEnd + 6)
                        return static_cast<Parameters>(
                            static_cast<std::size_t>(Parameters::DeltaX1)
                            + index - pbsEnd);

                    throw std::logic_error("Indexing Error");
                    return{};
                };

            for (std::size_t i = 0; i < n_paramters; ++i)
            {
                double* current_position = jacobi_row_out.ptr<double>(static_cast<int>(i));
                const Parameters param{ getCurrentParameter(radial_coefficients, tangential_coefficients, point_refinement, i) };

                double worker{};
                std::size_t index{};
                const double& b = (radial_coefficients) ? (*parameters.ptr<double>(index++)) : (0.0);
                const double& c = (radial_coefficients > 1) ? (*parameters.ptr<double>(index++)) : (0.0);
                const double& d = (radial_coefficients > 2) ? (*parameters.ptr<double>(index++)) : (0.0);

                const double& p1 = (tangential_coefficients) ? (*parameters.ptr<double>(index++)) : (0.0);
                const double& p2 = (tangential_coefficients) ? (*parameters.ptr<double>(index++)) : (0.0);

                const double& x_PBS = *parameters.ptr<double>(index++);
                const double& y_PBS = *parameters.ptr<double>(index++);

                for (auto it = indices.begin(); it != indices.end(); ++it)
                {
                    const bool sub{ it->first < 0 || it->second < 0 };
                    double contribution{};

                    // std::abs() because a negative first "index" declares subtraction
                    const cv::Point2d* firstIndex_raw{ points[static_cast<std::size_t>(std::abs(it->first))] };
                    const cv::Point2d* secondIndex_raw{ points[static_cast<std::size_t>(std::abs(it->second))] };

                    // Here also parameter but this is the point refinement 
                    const cv::Point2d* correction_first = (point_refinement) ?
                        point_correction[static_cast<std::size_t>(std::abs(it->first))] : &dummy;
                    const cv::Point2d* correction_second = (point_refinement) ?
                        point_correction[static_cast<std::size_t>(std::abs(it->second))] : &dummy;

                    const cv::Point2d firstIndex { correction_first->x + firstIndex_raw->x + x_PBS,
                        correction_first->y + firstIndex_raw->y + y_PBS };
                    const cv::Point2d secondIndex{ correction_second->x + secondIndex_raw->x + x_PBS, 
                        correction_second->y + secondIndex_raw->y + y_PBS };

                    const double& firstIndex_r2{ firstIndex.x * firstIndex.x + firstIndex.y * firstIndex.y };
                    const double& secondIndex_r2{ secondIndex.x * secondIndex.x + secondIndex.y * secondIndex.y };

                    double firstIndex_r4{}, firstIndex_r6{}, secondIndex_r4{}, secondIndex_r6{};

                    // Nested If, because the case of radial coefficients > 2 is considered rare. 
                    if (radial_coefficients > 1) {
                        firstIndex_r4 = firstIndex_r2 * firstIndex_r2;
                        secondIndex_r4 = secondIndex_r2 * secondIndex_r2;
                        if (radial_coefficients > 2) {
                            firstIndex_r6 = firstIndex_r2 * firstIndex_r4;
                            secondIndex_r6 = secondIndex_r2 * secondIndex_r4;
                        }
                    }

                    const double firstX =
                        firstIndex.x * (
                            1.0 + b * firstIndex_r2
                            + c * firstIndex_r4
                            + d * firstIndex_r6)
                        + 2.0 * p1 * firstIndex.x * firstIndex.y
                        + p2 * (
                            firstIndex_r2 + 2.0 * firstIndex.x * firstIndex.x);

                    const double secondY =
                        secondIndex.y * (
                            1.0 + b * secondIndex_r2
                            + c * secondIndex_r4
                            + d * secondIndex_r6)
                        + p1 * (
                            secondIndex_r2 + 2.0 * secondIndex.y * secondIndex.y)
                        + 2.0 * p2 * secondIndex.x * secondIndex.y;

                    // Corrected Coordinates are Saved 
                    corrected_pts.insert({ std::abs(it->first), firstX });
                    corrected_pts.insert({ std::abs(it->second) + 3, secondY });

                    // The magic begins here. 
                    switch (param) {
                    case(Parameters::k1):
                        contribution += std::get<0>(func_ptr)(firstIndex.x, firstIndex_r2) * secondY +
                            std::get<8>(func_ptr)(secondIndex.y, secondIndex_r2) * firstX;
                        break;
                    case(Parameters::k2):
                        contribution += std::get<1>(func_ptr)(firstIndex.x, firstIndex_r4) * secondY +
                            std::get<9>(func_ptr)(secondIndex.y, secondIndex_r4) * firstX;
                        break;
                    case(Parameters::k3):
                        contribution += std::get<2>(func_ptr)(firstIndex.x, firstIndex_r6) * secondY +
                            std::get<10>(func_ptr)(secondIndex.y, secondIndex_r6) * firstX;
                        break;
                    case(Parameters::P1):
                        contribution += std::get<3>(func_ptr)(firstIndex.x, firstIndex.y) * secondY +
                            std::get<11>(func_ptr)(secondIndex_r2, secondIndex.y * secondIndex.y) * firstX;
                        break;
                    case(Parameters::P2):
                        contribution += std::get<4>(func_ptr)(firstIndex_r2, firstIndex.x * firstIndex.x) * secondY +
                            std::get<12>(func_ptr)(secondIndex.x, secondIndex.y) * firstX;
                        break;
                    case(Parameters::X_PBS):
                        contribution +=
                            std::get<5>(func_ptr)(
                                firstIndex.x, firstIndex.y, b, c, d, p1, p2,
                                firstIndex_r2, firstIndex_r4)
                            * secondY +
                            std::get<17>(func_ptr)(
                                secondIndex.x, secondIndex.y, b, c, d, p1, p2,
                                secondIndex_r2, secondIndex_r4)
                            * firstX;
                        break;
                    case(Parameters::Y_PBS):
                        contribution +=
                            std::get<16>(func_ptr)(
                                firstIndex.x, firstIndex.y, b, c, d, p1, p2,
                                firstIndex_r2, firstIndex_r4)
                            * secondY +
                            std::get<13>(func_ptr)(
                                secondIndex.y, secondIndex.x, b, c, d, p2, p1,
                                secondIndex_r2, secondIndex_r4)
                            * firstX;
                        break;
                    case Parameters::DeltaX1:
                    case Parameters::DeltaY1:
                    case Parameters::DeltaX2:
                    case Parameters::DeltaY2:
                    case Parameters::DeltaX3:
                    case Parameters::DeltaY3:
                    {
                        // Enum order: DeltaX1, DeltaY1, DeltaX2, DeltaY2, ...
                        const int offset =
                            static_cast<int>(param)
                            - static_cast<int>(Parameters::DeltaX1);

                        const int point = offset / 2;         // 0, 1, or 2
                        const bool afterX = offset % 2 == 0;  // DeltaX or DeltaY

                        const int firstPoint = std::abs(it->first);
                        const int secondPoint = std::abs(it->second);

                        // Mapped coordinates: the undifferentiated factors
                        // in the product rule for X_first * Y_second.
                        

                        if (firstPoint == point)
                        {
                            const double derivative = afterX
                                ? std::get<5>(func_ptr)(   // dX/dx
                                    firstIndex.x, firstIndex.y,
                                    b, c, d, p1, p2,
                                    firstIndex_r2, firstIndex_r4)
                                : std::get<16>(func_ptr)(  // dX/dy
                                    firstIndex.x, firstIndex.y,
                                    b, c, d, p1, p2,
                                    firstIndex_r2, firstIndex_r4);

                            contribution += derivative * secondY;
                        }

                        if (secondPoint == point)
                        {
                            const double derivative = afterX
                                ? std::get<17>(func_ptr)(  // dY/dx: no swaps
                                    secondIndex.x, secondIndex.y,
                                    b, c, d, p1, p2,
                                    secondIndex_r2, secondIndex_r4)
                                : std::get<13>(func_ptr)(  // dY/dy: swap x/y AND p1/p2
                                    secondIndex.y, secondIndex.x,
                                    b, c, d, p2, p1,
                                    secondIndex_r2, secondIndex_r4);

                            contribution += firstX * derivative;
                        }
                        break;
                    }
                    }
                    worker += sub ? -contribution : contribution;
                }
                *current_position = worker;
            }
            assert(corrected_pts.size() == 6 && "There must be 6 saved Points in the Tree");

            residual = (corrected_pts[0] - corrected_pts[1]) * (corrected_pts[3] - corrected_pts[5]) -
                (corrected_pts[0] - corrected_pts[2]) * (corrected_pts[3] - corrected_pts[4]);

        }; 
    
    // Assemble the Jacobian and the actual nonlinear residuals.
    auto evaluate = [&build_Row](
        const cv::Mat& step_distortion,
        const cv::Mat& step_point_correct,
        cv::Mat& jacobian_parameter_out,
        cv::Mat& jacobian_point_out,
        cv::Mat& errors_out,
        const DistortionCalibrationConfig& config,
        const std::vector<std::array<const cv::Point2d*, 3>>& orderedPts,
        const std::map<const cv::Point2d*, std::size_t>& point_indices) -> bool
        {
            // Despite their names, these inputs are the CURRENT parameter
            // values and TOTAL point corrections, not iteration increments.
            const std::size_t radial{ config.n_radial_coefficients() };
            const std::size_t tangential{ config.n_tangential_coefficients() };
            const int n_params = static_cast<int>(radial + tangential + 2);
            const int n_lines = static_cast<int>(orderedPts.size());

            if (step_distortion.type() != CV_64F
                || step_distortion.cols != 1
                || step_distortion.rows != n_params)
                throw std::invalid_argument("Invalid distortion parameter vector");

            for (int col = 0; col < n_params; ++col)
                if (!std::isfinite(step_distortion.at<double>(col, 0)))
                    return false;

            jacobian_parameter_out.create(n_lines, n_params, CV_64F);
            errors_out.create(n_lines, 1, CV_64F);

            if (config.pointRefinement)
            {
                const int n_point_parameters =
                    static_cast<int>(2 * point_indices.size());

                if (step_point_correct.type() != CV_64F
                    || step_point_correct.cols != 1
                    || step_point_correct.rows != n_point_parameters)
                    throw std::invalid_argument("Invalid point correction vector");

                for (int col = 0; col < n_point_parameters; ++col)
                    if (!std::isfinite(step_point_correct.at<double>(col, 0)))
                        return false;

                // B has two columns per measured point. Repeated use of the
                // same point in different triples shares these columns.
                jacobian_point_out = cv::Mat::zeros(
                    n_lines, n_point_parameters, CV_64F);
            }
            else
                jacobian_point_out.release();

            cv::Mat worker(
                n_params + (config.pointRefinement ? 6 : 0), 1, CV_64F);

            for (std::size_t row = 0; row < orderedPts.size(); ++row)
            {
                double residual{};
                std::array<std::size_t, 3> indices{};
                std::array<cv::Point2d, 3> correction_values{};
                std::array<const cv::Point2d*, 3> correction{
                    nullptr, nullptr, nullptr
                };

                for (std::size_t point = 0; point < 3; ++point)
                {
                    if (!orderedPts[row][point])
                        throw std::invalid_argument("Null measured point");

                    if (config.pointRefinement)
                    {
                        const auto it = point_indices.find(orderedPts[row][point]);
                        if (it == point_indices.end()
                            || it->second >= point_indices.size())
                            throw std::logic_error("Missing or invalid point index");

                        indices[point] = it->second;
                        const int col = static_cast<int>(2 * indices[point]);
                        correction_values[point] = cv::Point2d(
                            step_point_correct.at<double>(col, 0),
                            step_point_correct.at<double>(col + 1, 0));
                        correction[point] = &correction_values[point];
                    }
                }

                build_Row(
                    radial,
                    tangential,
                    orderedPts[row],
                    correction,
                    step_distortion,
                    worker,
                    residual,
                    config.pointRefinement
                );

                if (!std::isfinite(residual))
                    return false;

                errors_out.at<double>(static_cast<int>(row), 0) = residual;

                for (int col = 0; col < n_params; ++col)
                {
                    const double value = worker.at<double>(col, 0);
                    if (!std::isfinite(value))
                        return false;

                    jacobian_parameter_out.at<double>(
                        static_cast<int>(row), col) = value;
                }

                if (config.pointRefinement)
                {
                    for (std::size_t point = 0; point < 3; ++point)
                    {
                        for (int coordinate = 0; coordinate < 2; ++coordinate)
                        {
                            const double value = worker.at<double>(
                                n_params + static_cast<int>(2 * point) + coordinate, 0);
                            if (!std::isfinite(value))
                                return false;

                            const int col =
                                static_cast<int>(2 * indices[point]) + coordinate;
                            jacobian_point_out.at<double>(
                                static_cast<int>(row), col) += value;
                        }
                    }
                }
            }

            return true;
        };

    // Solve the column-scaled, damped least-squares problem:
    //
    // [ a * S^-1       ] z = [ -residuals ]
    // [ sqrt(damping)I ]     [      0     ]
    //
    // step = S^-1 * z
    auto solve = [](const cv::Mat& a,
        const cv::Mat& residuals,
        const cv::Mat& scales,
        double damping,
        cv::Mat& step) -> bool
        {
            cv::Mat worker_opt = cv::Mat::zeros(
                a.rows + a.cols, a.cols, CV_64F);

            cv::Mat rhs = cv::Mat::zeros(
                a.rows + a.cols, 1, CV_64F);

            for (int row = 0; row < a.rows; ++row)
            {
                rhs.at<double>(row, 0) =
                    -residuals.at<double>(row, 0);

                for (int col = 0; col < a.cols; ++col)
                    worker_opt.at<double>(row, col) =
                    a.at<double>(row, col)
                    / scales.at<double>(col, 0);
            }

            for (int col = 0; col < a.cols; ++col)
                worker_opt.at<double>(a.rows + col, col) =
                std::sqrt(damping);

            cv::Mat scaled_step;

            if (!cv::solve(
                worker_opt, rhs, scaled_step, cv::DECOMP_SVD))
                return false;

            step.create(a.cols, 1, CV_64F);

            for (int col = 0; col < a.cols; ++col)
            {
                const double value =
                    scaled_step.at<double>(col, 0)
                    / scales.at<double>(col, 0);

                if (!std::isfinite(value))
                    return false;

                step.at<double>(col, 0) = value;
            }

            return true;
        };



    // --- Main function starts here ---
    if (!config.valid()) throw std::invalid_argument("Invalid config file");
    if (imagePts.empty())
        throw std::invalid_argument("No image points");
    std::size_t n_elements{};
    std::size_t n_lines{};

    // There can not be a line defined by less than 3 points
    if (std::any_of(imagePts.begin(), imagePts.end(),
        [](const std::vector<cv::Point2d>& line) -> bool {
            return line.size() < 3;
        })) throw std::invalid_argument("Line Container has less than 3 Points");

    // Number of global model parameters, including PBS.
    const int n_params = static_cast<int>(
        config.n_radial_coefficients() + config.n_tangential_coefficients() + 2);

    for (const auto& pt : imagePts) {
        std::size_t elements = pt.size();
        n_lines += std::max(static_cast<std::size_t>(elements - 2), static_cast<std::size_t>(0));
        n_elements += elements;
    }

    const std::size_t radial{ config.n_radial_coefficients() };
    const std::size_t tangential{ config.n_tangential_coefficients() };

    if (radial > 3 || (tangential != 0 && tangential != 2))
        throw std::invalid_argument("Invalid parameter count");

    if (n_lines < static_cast<std::size_t>(n_params))
        throw std::runtime_error("Not enough equations for optimization");

    cv::Mat a(
        static_cast<int>(n_lines),
        static_cast<int>(radial + tangential + 2),
        CV_64F, cv::Scalar(0.0));
    cv::Mat x(a.cols, 1, CV_64F, cv::Scalar(0.0)); // Parameter Vector Distortion

    cv::Mat x_temp; // Temporary parameter vector

    // Total corrections relative to the original measurements:
    // (dx_0, dy_0, dx_1, dy_1, ...).
    cv::Mat point_correct;
    if (config.pointRefinement)
        point_correct = cv::Mat::zeros(
            static_cast<int>(2 * n_elements), 1, CV_64F);

    cv::Mat b; // Global point Jacobian; empty without point refinement.

    cv::Mat residuals;
    cv::Mat residuals_after;
    cv::Mat a_temp;
    cv::Mat step;
    cv::Mat scales(a.cols, 1, CV_64F);

    // Separate copies in different input lines are separate measurements.
    std::map<const cv::Point2d*, std::size_t> point_indices;
    std::size_t point_index{ 0 };
    for (const auto& pts_on_line : imagePts)
        for (const auto& point : pts_on_line)
            point_indices.emplace(&point, point_index++);

    std::vector<std::array<const cv::Point2d*, 3>> orderedPts;
    orderedPts.reserve(n_lines);

    for (const auto& pts_on_line : imagePts)
    {
        for (std::size_t offset = 1;
            offset + 1 < pts_on_line.size();
            ++offset)
        {
            orderedPts.push_back({
                &pts_on_line.front(),
                &pts_on_line[offset],
                &pts_on_line.back()
                });
        }
    }

    std::size_t iterations{ 0 };

    // This lambda owns the existing LM loop without point refinement.
    // The future refinement solver will use the same evaluate lambda.
    auto solve_without_point_refinement = [&]() -> void
        {
            const std::size_t max_iterations{ 1000 };
            const int max_retries{ 30 };

            // Tolerances for normalized image coordinates.
            const double residual_tolerance{ 1e-12 };
            const double gradient_tolerance{ 1e-10 };
            const double min_damping{ 1e-15 };
            const double max_damping{ 1e16 };

            double damping{ 1e-3 };
            bool converged{ false };

            if (!evaluate(x, point_correct, a, b, residuals, config, orderedPts, point_indices))
                throw std::runtime_error("Invalid initial model evaluation");

            double cost = 0.5 * residuals.dot(residuals);

            if (!std::isfinite(cost))
                throw std::runtime_error("Invalid initial objective");

            while (true)
            {
                if (std::sqrt(2.0 * cost / a.rows) <= residual_tolerance)
                {
                    converged = true;
                    break;
                }

                const cv::Mat gradient = a.t() * residuals;
                double scaled_gradient_max{ 0.0 };

                // Apply Scaling with respect to the columns of a
                for (int col = 0; col < a.cols; ++col)
                {
                    double scale = cv::norm(a.col(col), cv::NORM_L2);

                    if (!std::isfinite(scale))
                        throw std::runtime_error("Invalid Jacobian column norm");

                    // PBS derivatives can vanish at zero distortion.
                    if (scale < 1e-12)
                        scale = 1.0;

                    scales.at<double>(col, 0) = scale;

                    const double value =
                        gradient.at<double>(col, 0) / scale;

                    if (!std::isfinite(value))
                        throw std::runtime_error("Invalid objective gradient");

                    scaled_gradient_max =
                        std::max(scaled_gradient_max, std::abs(value));
                }

                if (scaled_gradient_max <= gradient_tolerance)
                {
                    converged = true;
                    break;
                }

                if (iterations >= max_iterations)
                    break;

                bool accepted{ false };

                for (int retry = 0;
                    retry < max_retries && iterations < max_iterations;
                    ++retry)
                {
                    ++iterations;

                    if (solve(a, residuals, scales, damping, step))
                    {
                        x_temp = x + step;

                        if (evaluate(x_temp, point_correct, a_temp, b, residuals_after, config, orderedPts, point_indices))
                        {
                            const double cost_after =
                                0.5 * residuals_after.dot(residuals_after);

                            const cv::Mat linear_change = a * step;

                            const double predicted_reduction =
                                -gradient.dot(step)
                                - 0.5 * linear_change.dot(linear_change);

                            const double actual_reduction = cost - cost_after;

                            if (std::isfinite(cost_after)
                                && std::isfinite(predicted_reduction)
                                && predicted_reduction > 0.0
                                && actual_reduction > 0.0)
                            {
                                const double ratio =
                                    actual_reduction / predicted_reduction;

                                if (std::isfinite(ratio) && ratio > 1e-4)
                                {
                                    // Accept the trial without sharing its buffers.
                                    std::swap(x, x_temp);
                                    std::swap(a, a_temp);
                                    std::swap(residuals, residuals_after);

                                    cost = cost_after;
                                    accepted = true;

                                    if (ratio > 0.75)
                                        damping = std::max(
                                            min_damping, damping / 3.0);
                                    else if (ratio < 0.25)
                                        damping = std::min(
                                            max_damping, damping * 2.0);

                                    break;
                                }
                            }
                        }
                    }// Reject the trial; keep the current parameters.

                if (damping >= max_damping)
                    break;

                damping = std::min(max_damping, damping * 10.0);
            }

            if (!accepted)
                throw std::runtime_error(
                    "LM could not find an improving step");
        }

        if (!converged)
            throw std::runtime_error(
                "LM iteration limit reached without convergence");
    };

    if (config.pointRefinement)
    {
        // Future solver: use evaluate(x, point_correct, a, b, residuals,
        //                            config, orderedPts, point_indices).
        // It must own its constrained iteration and step acceptance.
        throw std::logic_error("Point refinement solver is not implemented yet");
    }
    else
        solve_without_point_refinement();

    std::size_t index{ 0 };

    DistortionCalibrationResult result{};
    result.k1 = (radial) ? (x.at<double>(cv::Point(0, index++))) : (std::numeric_limits<double>::quiet_NaN());
    result.k2 = (radial > 1) ? (x.at<double>(cv::Point(0, index++))) : (std::numeric_limits<double>::quiet_NaN());
    result.k3 = (radial > 2) ? (x.at<double>(cv::Point(0, index++))) : (std::numeric_limits<double>::quiet_NaN());
    result.p1 = (tangential) ? (x.at<double>(cv::Point(0, index++))) : (std::numeric_limits<double>::quiet_NaN());
    result.p2 = (tangential) ? (x.at<double>(cv::Point(0, index++))) : (std::numeric_limits<double>::quiet_NaN());
    result.x_PBS = x.at<double>(cv::Point(0, index++));
    result.y_PBS = x.at<double>(cv::Point(0, index++));

    result.iterations = iterations;
    // result.point_corrections = (config.pointRefinement) ? ()

    return result;
}


void CameraCalibration::_TestDistortionCalibration(
    const _TestDistortionCalibrationConfig& config_test,
    const DistortionCalibrationConfig& config,
    const std::filesystem::path& path)
{
    if (!config.valid())
        throw std::invalid_argument(
            "Use a valid configuration");
    if (!config_test.valid())
        throw std::invalid_argument(
        "Use a valid configuration");

    const auto radial = config.n_radial_coefficients();
    const auto tangential = config.n_tangential_coefficients();

    const double& k1 = config_test.k1;
    const double& k2 = config_test.k2;
    const double& k3 = config_test.k3;
    const double& p1 = config_test.p1;
    const double& p2 = config_test.p2;

    // Identity K: all coordinates are already normalized.
    const cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat coefficients = (cv::Mat_<double>(1, 5) <<
        k1, k2, p1, p2, k3);

    const cv::Vec3d zero(0.0, 0.0, 0.0);
    const std::size_t n_lines = config_test.lineParameters.angles.size() * config_test.lineParameters.offset.size();
    const std::size_t n_points = n_lines * config_test.lineParameters.points_on_line;

    std::vector<std::vector<cv::Point2d>> points{}; // initialization only for the first dimension
    points.reserve(n_lines);

    double generation_error = 0.0;

    cv::RNG rng{ config_test.seed_Random };

    // Four directions, four offsets: 16 straight reference lines.
    for (const double angle : config_test.lineParameters.angles)
    {
        const cv::Point2d direction(std::cos(angle), std::sin(angle));
        const cv::Point2d normal(-direction.y, direction.x);

        for (const double& offset : config_test.lineParameters.offset)
        {
            std::vector<cv::Point2d> straight;

            straight.reserve(config_test.lineParameters.points_on_line);

            for (int i = 0; i < config_test.lineParameters.points_on_line; ++i)
            {
                const double range{ config_test.lineParameters.image_boundaries.second - config_test.lineParameters.image_boundaries.first };
                const double t = range * static_cast<double>(i) / 
                    config_test.lineParameters.points_on_line - (range/2.0);

                straight.emplace_back(
                    offset * normal.x + t * direction.x,
                    offset * normal.y + t * direction.y);
            }

            std::vector<cv::Point2d> meassured_distorted{};

            // Ironcially undistortPoints does a distortion of the image coordiantes 
            cv::undistortPoints(
                straight,
                meassured_distorted,
                cameraMatrix,
                coefficients,
                cv::noArray(),
                cv::noArray(),
                cv::TermCriteria(
                    cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                    100, 1e-14)
            );
            
            if (config_test.add_noise) {
                const double noise_norm = config_test.stddev_noise * 2.0 / 2464.0;

                for (auto& pt : meassured_distorted) {
                    cv::Point2d noise{
                        rng.gaussian(noise_norm),
                        rng.gaussian(noise_norm)
                    };
                    pt += noise;
                }
            }

            points.push_back(std::move(meassured_distorted));
        }
    }

    const auto result = DistortionCalibration(points, config);

    auto print = [](const char* name, double expected, double estimated)
        {
            std::cout << name
                << ": expected = " << expected
                << ", estimated = " << estimated
                << ", difference = " << estimated - expected << '\n';
        };

    std::cout.precision(12);
    std::cout << "Maximum data-generation error: "
        << generation_error << '\n';

    if (radial >= 1) print("k1", k1, result.k1);
    if (radial >= 2) print("k2", k2, result.k2);
    if (radial >= 3) print("k3", k3, result.k3);

    if (tangential)
    {
        print("p1", p1, result.p1);
        print("p2", p2, result.p2);
    }

    print("x_PBS", 0.0, result.x_PBS);
    print("y_PBS", 0.0, result.y_PBS);

    std::cout << "Iterations: " << result.iterations << '\n';

    if (config_test.save)
        config_test.save_to_file(
            path,
            result,
            false
        );

    return;
}

