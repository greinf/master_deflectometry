#ifndef CAMERA_CALIBRATION_HPP
#define CAMERA_CALIBRATION_HPP

#include <string>
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

struct CalibrationConfig {
    cv::Size boardSize{};
    float squareSize = 0.0f;

    // Intrinsic initialization
    bool useIntrinsicGuess = false;

    // Principal point
    bool fixPrincipalPoint = false;
    bool principalPointAtImageCenter = true;
    cv::Point2d principalPointGuess{}; // only used if principalPointAtImageCenter == false

    // Lens model
    bool zeroTangentDist = false;

    bool fixAspectRatio = false;
    float aspectRatio = 1.0f;

    bool fixK1 = false;
    bool fixK2 = false;
    bool fixK3 = false;
    bool fixK4 = false;
    bool fixK5 = false;
    bool fixK6 = false;

    // Stereo
    bool fixIntrinsicsInStereo = true;

    bool writeExtrinsics = true;
    bool writePerViewErrors = true;

    std::string outputFileName;

    [[nodiscard]] bool isValid() const;
    [[nodiscard]] int makeCalibrationFlags() const;
};

struct MonoCalibrationResult {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    std::vector<float> perViewErrors;
    double avgReprojectionError = -1.0;
    cv::Size imageSize;
    bool success = false;
};

struct StereoCalibrationResult {
    MonoCalibrationResult left;
    MonoCalibrationResult right;

    cv::Mat R, T, E, F;
    double stereoError = -1.0;
    bool success = false;
};

struct StereoImagePair {
    cv::Mat left;
    cv::Mat right;
};

class CameraCalibration {
public:
    CameraCalibration() = default;

    [[nodiscard]] MonoCalibrationResult calibrateMono(
        const std::vector<cv::Mat>& images,
        const CalibrationConfig& config,
        bool drawDetectedCorners = false,
        int subPixWindow = 11) const;

    [[nodiscard]] StereoCalibrationResult calibrateStereo(
        const std::vector<StereoImagePair>& imagePairs,
        const CalibrationConfig& config,
        bool drawDetectedCorners = false,
        int subPixWindow = 11) const;

    bool saveMonoCalibration(
        const std::string& filename,
        const CalibrationConfig& config,
        const MonoCalibrationResult& result) const;

    bool saveStereoCalibration(
        const std::string& filename,
        const CalibrationConfig& config,
        const StereoCalibrationResult& result) const;

private:
    [[nodiscard]] static bool detectChessboardCorners(
        const cv::Mat& image,
        const cv::Size& boardSize,
        std::vector<cv::Point2f>& corners,
        int subPixWindow);

    [[nodiscard]] static std::vector<cv::Point3f> createBoardObjectPoints(
        const cv::Size& boardSize,
        float squareSize);

    cv::Mat createInitialCameraMatrix(
        const cv::Size& imageSize,
        const CalibrationConfig& config) const;

    [[nodiscard]] static double computeReprojectionErrors(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs,
        std::vector<float>& perViewErrors);
};

#endif