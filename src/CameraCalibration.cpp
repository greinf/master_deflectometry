#include "CameraCalibration.hpp"

#include <iostream>
#include <ctime>
#include <sstream>

bool CalibrationConfig::isValid() const {
    if (boardSize.width <= 0 || boardSize.height <= 0) {
        return false;
    }
    if (squareSize <= 0.0f) {
        return false;
    }
    return true;
}

int CalibrationConfig::makeCalibrationFlags() const {
    int flags = 0;

    if (fixPrincipalPoint) flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if (zeroTangentDist)   flags |= cv::CALIB_ZERO_TANGENT_DIST;
    if (fixAspectRatio)    flags |= cv::CALIB_FIX_ASPECT_RATIO;
    if (fixK1)             flags |= cv::CALIB_FIX_K1;
    if (fixK2)             flags |= cv::CALIB_FIX_K2;
    if (fixK3)             flags |= cv::CALIB_FIX_K3;
    if (fixK4)             flags |= cv::CALIB_FIX_K4;
    if (fixK5)             flags |= cv::CALIB_FIX_K5;

    return flags;
}

bool CameraCalibration::detectChessboardCorners(
    const cv::Mat& image,
    const cv::Size& boardSize,
    std::vector<cv::Point2f>& corners,
    int subPixWindow)
{
    if (image.empty()) {
        return false;
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else {
        gray = image;
    }

    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK;

    const bool found = cv::findChessboardCorners(gray, boardSize, corners, flags);

    if (!found) {
        return false;
    }

    cv::cornerSubPix(
        gray,
        corners,
        cv::Size(subPixWindow, subPixWindow),
        cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 1e-4));

    return true;
}

std::vector<cv::Point3f> CameraCalibration::createBoardObjectPoints(
    const cv::Size& boardSize,
    float squareSize)
{
    std::vector<cv::Point3f> objectPoints;
    objectPoints.reserve(static_cast<size_t>(boardSize.width * boardSize.height));

    for (int y = 0; y < boardSize.height; ++y) {
        for (int x = 0; x < boardSize.width; ++x) {
            objectPoints.emplace_back(
                static_cast<float>(x) * squareSize,
                static_cast<float>(y) * squareSize,
                0.0f);
        }
    }

    return objectPoints;
}

double CameraCalibration::computeReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    std::vector<float>& perViewErrors)
{
    std::vector<cv::Point2f> projectedPoints;
    perViewErrors.resize(objectPoints.size());

    double totalErr2 = 0.0;
    size_t totalPoints = 0;

    for (size_t i = 0; i < objectPoints.size(); ++i) {
        cv::projectPoints(
            objectPoints[i],
            rvecs[i],
            tvecs[i],
            cameraMatrix,
            distCoeffs,
            projectedPoints);

        const double err = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
        const size_t n = objectPoints[i].size();

        perViewErrors[i] = static_cast<float>(std::sqrt((err * err) / static_cast<double>(n)));
        totalErr2 += err * err;
        totalPoints += n;
    }

    if (totalPoints == 0) {
        return -1.0;
    }

    return std::sqrt(totalErr2 / static_cast<double>(totalPoints));
}

MonoCalibrationResult CameraCalibration::calibrateMono(
    const std::vector<cv::Mat>& images,
    const CalibrationConfig& config,
    bool drawDetectedCorners,
    int subPixWindow) const
{
    MonoCalibrationResult result{};

    if (!config.isValid()) {
        std::cerr << "CalibrationConfig is invalid.\n";
        return result;
    }

    if (images.empty()) {
        std::cerr << "No images provided for mono calibration.\n";
        return result;
    }

    std::vector<std::vector<cv::Point2f>> imagePoints;
    imagePoints.reserve(images.size());

    cv::Size imageSize{};

    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Mat& image = images[i];

        if (image.empty()) {
            continue;
        }

        if (imageSize.empty()) {
            imageSize = image.size();
        }
        else if (image.size() != imageSize) {
            std::cerr << "Skipping image with inconsistent size at index " << i << ".\n";
            continue;
        }

        std::vector<cv::Point2f> corners;
        const bool found = detectChessboardCorners(image, config.boardSize, corners, subPixWindow);

        if (!found) {
            continue;
        }

        imagePoints.push_back(corners);

        if (drawDetectedCorners) {
            cv::Mat vis = image.clone();
            cv::drawChessboardCorners(vis, config.boardSize, corners, true);
            cv::imshow("Mono calibration corners", vis);
            cv::waitKey(50);
            cv::destroyWindow("Mono calibration corners");
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "No valid chessboard detections found for mono calibration.\n";
        return result;
    }

    const std::vector<cv::Point3f> boardPoints =
        createBoardObjectPoints(config.boardSize, config.squareSize);

    std::vector<std::vector<cv::Point3f>> objectPoints(imagePoints.size(), boardPoints);

    result.imageSize = imageSize;
    result.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    if (config.fixAspectRatio) {
        result.cameraMatrix.at<double>(0, 0) = config.aspectRatio;
    }

    result.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    const int flags = config.makeCalibrationFlags();

    cv::calibrateCamera(
        objectPoints,
        imagePoints,
        imageSize,
        result.cameraMatrix,
        result.distCoeffs,
        result.rvecs,
        result.tvecs,
        flags);

    result.avgReprojectionError = computeReprojectionErrors(
        objectPoints,
        imagePoints,
        result.rvecs,
        result.tvecs,
        result.cameraMatrix,
        result.distCoeffs,
        result.perViewErrors);

    result.success = cv::checkRange(result.cameraMatrix) && cv::checkRange(result.distCoeffs);

    saveMonoCalibration(config.outputFileName, config, result);

    return result;
}

StereoCalibrationResult CameraCalibration::calibrateStereo(
    const std::vector<StereoImagePair>& imagePairs,
    const CalibrationConfig& config,
    bool drawDetectedCorners,
    int subPixWindow) const
{
    StereoCalibrationResult result{};

    if (!config.isValid()) {
        std::cerr << "CalibrationConfig is invalid.\n";
        return result;
    }

    if (imagePairs.empty()) {
        std::cerr << "No stereo image pairs provided.\n";
        return result;
    }

    std::vector<std::vector<cv::Point2f>> leftImagePoints;
    std::vector<std::vector<cv::Point2f>> rightImagePoints;

    leftImagePoints.reserve(imagePairs.size());
    rightImagePoints.reserve(imagePairs.size());

    cv::Size imageSize{};

    for (size_t i = 0; i < imagePairs.size(); ++i) {
        const cv::Mat& left = imagePairs[i].left;
        const cv::Mat& right = imagePairs[i].right;

        if (left.empty() || right.empty()) {
            continue;
        }

        if (left.size() != right.size()) {
            std::cerr << "Skipping stereo pair with mismatched image sizes at index " << i << ".\n";
            continue;
        }

        if (imageSize.empty()) {
            imageSize = left.size();
        }
        else if (left.size() != imageSize) {
            std::cerr << "Skipping stereo pair with inconsistent size at index " << i << ".\n";
            continue;
        }

        std::vector<cv::Point2f> leftCorners;
        std::vector<cv::Point2f> rightCorners;

        const bool foundLeft = detectChessboardCorners(left, config.boardSize, leftCorners, subPixWindow);
        const bool foundRight = detectChessboardCorners(right, config.boardSize, rightCorners, subPixWindow);

        if (!(foundLeft && foundRight)) {
            continue;
        }

        leftImagePoints.push_back(leftCorners);
        rightImagePoints.push_back(rightCorners);

        if (drawDetectedCorners) {
            cv::Mat leftVis = left.clone();
            cv::Mat rightVis = right.clone();

            cv::drawChessboardCorners(leftVis, config.boardSize, leftCorners, true);
            cv::drawChessboardCorners(rightVis, config.boardSize, rightCorners, true);

            cv::imshow("Stereo left corners", leftVis);
            cv::imshow("Stereo right corners", rightVis);
            cv::waitKey(50);
        }
    }

    if (leftImagePoints.empty() || rightImagePoints.empty()) {
        std::cerr << "No valid stereo chessboard detections found.\n";
        return result;
    }

    const std::vector<cv::Point3f> boardPoints =
        createBoardObjectPoints(config.boardSize, config.squareSize);

    std::vector<std::vector<cv::Point3f>> objectPoints(leftImagePoints.size(), boardPoints);

    result.left.imageSize = imageSize;
    result.right.imageSize = imageSize;

    result.left.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    result.right.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    if (config.fixAspectRatio) {
        result.left.cameraMatrix.at<double>(0, 0) = config.aspectRatio;
        result.right.cameraMatrix.at<double>(0, 0) = config.aspectRatio;
    }

    result.left.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    result.right.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    const int flags = config.makeCalibrationFlags();

    cv::calibrateCamera(
        objectPoints,
        leftImagePoints,
        imageSize,
        result.left.cameraMatrix,
        result.left.distCoeffs,
        result.left.rvecs,
        result.left.tvecs,
        flags);

    cv::calibrateCamera(
        objectPoints,
        rightImagePoints,
        imageSize,
        result.right.cameraMatrix,
        result.right.distCoeffs,
        result.right.rvecs,
        result.right.tvecs,
        flags);

    result.left.avgReprojectionError = computeReprojectionErrors(
        objectPoints,
        leftImagePoints,
        result.left.rvecs,
        result.left.tvecs,
        result.left.cameraMatrix,
        result.left.distCoeffs,
        result.left.perViewErrors);

    result.right.avgReprojectionError = computeReprojectionErrors(
        objectPoints,
        rightImagePoints,
        result.right.rvecs,
        result.right.tvecs,
        result.right.cameraMatrix,
        result.right.distCoeffs,
        result.right.perViewErrors);

    result.stereoError = cv::stereoCalibrate(
        objectPoints,
        leftImagePoints,
        rightImagePoints,
        result.left.cameraMatrix,
        result.left.distCoeffs,
        result.right.cameraMatrix,
        result.right.distCoeffs,
        imageSize,
        result.R,
        result.T,
        result.E,
        result.F,
        cv::CALIB_FIX_INTRINSIC);

    result.left.success =
        cv::checkRange(result.left.cameraMatrix) &&
        cv::checkRange(result.left.distCoeffs);

    result.right.success =
        cv::checkRange(result.right.cameraMatrix) &&
        cv::checkRange(result.right.distCoeffs);

    result.success = result.left.success && result.right.success &&
        cv::checkRange(result.R) && cv::checkRange(result.T);

    saveStereoCalibration(config.outputFileName, config, result);

    return result;
}

bool CameraCalibration::saveMonoCalibration(
    const std::string& filename,
    const CalibrationConfig& config,
    const MonoCalibrationResult& result) const
{
    if (!result.success) {
        return false;
    }

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }

    std::time_t now = std::time(nullptr);
    char timeBuffer[128]{};
#if defined(_WIN32)
    std::tm tmLocal{};
    localtime_s(&tmLocal, &now);
    std::strftime(timeBuffer, sizeof(timeBuffer), "%c", &tmLocal);
#else
    std::tm* tmLocal = std::localtime(&now);
    std::strftime(timeBuffer, sizeof(timeBuffer), "%c", tmLocal);
#endif

    fs << "calibration_time" << timeBuffer;
    fs << "image_width" << result.imageSize.width;
    fs << "image_height" << result.imageSize.height;
    fs << "board_width" << config.boardSize.width;
    fs << "board_height" << config.boardSize.height;
    fs << "square_size" << config.squareSize;

    fs << "camera_matrix" << result.cameraMatrix;
    fs << "distortion_coefficients" << result.distCoeffs;
    fs << "avg_reprojection_error" << result.avgReprojectionError;

    if (config.writePerViewErrors && !result.perViewErrors.empty()) {
        fs << "per_view_reprojection_errors" << cv::Mat(result.perViewErrors);
    }

    if (config.writeExtrinsics && !result.rvecs.empty() && !result.tvecs.empty()) {
        cv::Mat extrinsics(static_cast<int>(result.rvecs.size()), 6, CV_64F);

        for (size_t i = 0; i < result.rvecs.size(); ++i) {
            cv::Mat r = extrinsics(cv::Range(static_cast<int>(i), static_cast<int>(i + 1)), cv::Range(0, 3));
            cv::Mat t = extrinsics(cv::Range(static_cast<int>(i), static_cast<int>(i + 1)), cv::Range(3, 6));

            result.rvecs[i].reshape(1, 1).copyTo(r);
            result.tvecs[i].reshape(1, 1).copyTo(t);
        }

        fs << "extrinsic_parameters" << extrinsics;
    }

    fs.release();
    return true;
}

bool CameraCalibration::saveStereoCalibration(
    const std::string& filename,
    const CalibrationConfig& config,
    const StereoCalibrationResult& result) const
{
    if (!result.success) {
        return false;
    }

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }

    std::time_t now = std::time(nullptr);
    char timeBuffer[128]{};
#if defined(_WIN32)
    std::tm tmLocal{};
    localtime_s(&tmLocal, &now);
    std::strftime(timeBuffer, sizeof(timeBuffer), "%c", &tmLocal);
#else
    std::tm* tmLocal = std::localtime(&now);
    std::strftime(timeBuffer, sizeof(timeBuffer), "%c", tmLocal);
#endif

    fs << "calibration_time" << timeBuffer;
    fs << "image_width" << result.left.imageSize.width;
    fs << "image_height" << result.left.imageSize.height;
    fs << "board_width" << config.boardSize.width;
    fs << "board_height" << config.boardSize.height;
    fs << "square_size" << config.squareSize;

    fs << "camera_matrix1" << result.left.cameraMatrix;
    fs << "distortion_coefficients1" << result.left.distCoeffs;
    fs << "camera_matrix2" << result.right.cameraMatrix;
    fs << "distortion_coefficients2" << result.right.distCoeffs;

    fs << "avg_reprojection_error_cam1" << result.left.avgReprojectionError;
    fs << "avg_reprojection_error_cam2" << result.right.avgReprojectionError;
    fs << "stereo_error" << result.stereoError;

    if (config.writePerViewErrors && !result.left.perViewErrors.empty()) {
        fs << "per_view_reprojection_errors_cam1" << cv::Mat(result.left.perViewErrors);
    }
    if (config.writePerViewErrors && !result.right.perViewErrors.empty()) {
        fs << "per_view_reprojection_errors_cam2" << cv::Mat(result.right.perViewErrors);
    }

    fs << "R" << result.R;
    fs << "T" << result.T;
    fs << "E" << result.E;
    fs << "F" << result.F;

    fs.release();
    return true;
}