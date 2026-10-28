#ifndef CAMERA_CALIBRATION_HPP
#define CAMERA_CALIBRATION_HPP

#include <string>
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>

struct CalibrationConfig {
    cv::Size boardSize{11, 8};
    float squareSize = 30.0f;

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

    // Five-coefficient Brown model; k4-k6 are deliberately not enabled.
    bool useSBDetector = true; // preserve the original detector by default
    int subPixWindow = 11;      // half window, only for the classic detector
    int preprocessingKernelBias = 2; // -1: box integration only, no Gaussian
    int minimumViews = 3;       // structural minimum, not a quality guarantee
    int maxIterations = 100;
    double epsilon = 1e-10;
    double initialFocalLengthPx = 14800.0; // generic image API with useIntrinsicGuess; configurable

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
    int calibrationFlags = 0;
    double optimizerRms = -1.0;
    std::vector<int> acceptedImageIndices;
    std::vector<int> rejectedImageIndices;
    std::string errorMessage;
};

struct DistortionCalibrationConfig {
    bool fixK1 = false;
    bool fixK2 = false;
    bool fixK3 = true;
    bool fixTangent = false;

    [[nodiscard]] bool valid() const {
        if (fixK1 == true && fixK2 == false || fixK3 == false)
            return false;
        if (fixK2 == true || fixK3 == false)
            return false;
        if (fixK1 && fixK2 && fixK3 && !fixTangent) {
            std::cout << "WARNING: ONLY TANGENTIAL DISTORTION ESTIMATION " << std::endl;
        }
        return true;
    }
};

struct DistortionCalibrationResult {
    double k1{};
    double k2{};
    double k3{};
    double p1{};
    double p2{};
private:
    double iterations{};
    std::vector<cv::Vec2d> point_corrections{};

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

    [[nodiscard]] static DistortionCalibrationResult DistortionCalibration(
        const std::vector<std::vector<cv::Point2d>>& imagePts,
        const DistortionCalibrationConfig& config);

    [[nodiscard]] static MonoCalibrationResult calibrateMonoSyn(
        const std::vector<std::vector<cv::Point2f>>& chessboardCorners,
        const CalibrationConfig& config,
        const cv::Mat& intrinsic_guess,
        const cv::Mat& dist_coeffs_guess,
        const cv::Size& imgage_size = { 2464, 2056 },
        bool drawDetectedCorners = false);

    [[nodiscard]] MonoCalibrationResult calibrateMono(
        const std::vector<cv::Mat>& images,
        const CalibrationConfig& config,
        bool drawDetectedCorners = false,
        int subPixWindow = -1) const; // -1 uses config.subPixWindow

    [[nodiscard]] StereoCalibrationResult calibrateStereo(
        const std::vector<StereoImagePair>& imagePairs,
        const CalibrationConfig& config,
        bool drawDetectedCorners = false,
        int subPixWindow = -1) const;

    bool saveMonoCalibration(
        const std::string& filename,
        const CalibrationConfig& config,
        const MonoCalibrationResult& result) const;

    bool saveStereoCalibration(
        const std::string& filename,
        const CalibrationConfig& config,
        const StereoCalibrationResult& result) const;

    [[nodiscard]] static bool detectChessboardCorners(
        const cv::Mat& image,
        const cv::Size& boardSize,
        std::vector<cv::Point2f>& corners,
        int subPixWindow = 11,
        bool useSBDetector = false);

    [[nodiscard]] static cv::Mat toGray8(const cv::Mat& image);

    static void addNoise(
        std::vector<std::vector<cv::Point2f>>& chessboardCorners,
        std::pair<double, double> noise_boundaries,
        std::uint32_t seed = 42U)
    {
        if (!std::isfinite(noise_boundaries.first) ||
            !std::isfinite(noise_boundaries.second) ||
            noise_boundaries.first > noise_boundaries.second)
            throw std::invalid_argument("Invalid noise interval");
        auto dist = std::uniform_real_distribution(noise_boundaries.first, noise_boundaries.second);
        auto generator = std::mt19937(seed);

        for (auto& corner_batch : chessboardCorners) {
            for (auto& corner : corner_batch) {
                corner += cv::Point2f{
                    static_cast<float>(dist(generator)),
                    static_cast<float>(dist(generator))
                };
            }
        }
        return;
    }

public:
    [[nodiscard]] static std::vector<cv::Point3f> createBoardObjectPoints(
        const cv::Size& boardSize,
        float squareSize);

    static cv::Mat createInitialCameraMatrix(
        const cv::Size& imageSize,
        const CalibrationConfig& config);

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
