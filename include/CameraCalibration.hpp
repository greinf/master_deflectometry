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
#include <filesystem>
#include <optional>
#include <fstream>
#include <ctime>

struct CalibrationConfig {
    cv::Size boardSize{ 11, 8 };
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
    bool pointRefinement = false;

    [[nodiscard]] bool valid() const {
        if (fixK1 == true && (fixK2 == false || fixK3 == false))
            return false;
        if (fixK2 == true || fixK3 == false)
            return false;
        if (fixK1 && fixK2 && fixK3 && !fixTangent) {
            std::cout << "WARNING: ONLY TANGENTIAL DISTORTION ESTIMATION " << std::endl;
        }
        return true;
    }

    [[nodiscard]] std::size_t n_radial_coefficients() const {
        if (!valid()) throw std::runtime_error("Invalid distortion Calib Config");
        if (!fixK3) return 3;
        if (!fixK2) return 2;
        if (!fixK1) return 1;
        if (fixK1) return 0;
    }

    [[nodiscard]] std::size_t n_tangential_coefficients() const {
        if (fixTangent) return 0;
        return 2;
    }

    [[nodiscard]] int n_parameters() const {
        int n{};
        if (!fixK1) ++n;
        if (!fixK2) ++n;
        if (!fixK3) ++n;
        if (fixTangent) n += 2;
        if (pointRefinement) n += 6;
        return n;
    }
};


struct DistortionCalibrationResult {
    // Result of iterative algorithmus
    double k1{};
    double k2{};
    double k3{};
    double p1{};
    double p2{};
    double x_PBS{};
    double y_PBS{};

    int iterations{};
    std::vector<cv::Point2d> point_corrections{};    
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

    struct _TestDistortionCalibrationConfig {
        // Distortion Parameters
        double k1{0.15};
        double k2{-0.2};
        double k3{0.1};
        double p1{0.2};
        double p2{-0.2};
        bool add_noise{ true };
        bool save{ true };

        std::size_t pixel_x{ 2464 };

        // Noise Charakteristics 
        double stddev_noise{ 0.0 };
        std::uint64_t seed_Random{ 0 };
       

        struct LineParameters {
            std::size_t points_on_line{4};
            // Angles that define the direction of the line. angle 0 is x axis
            std::vector<double> angles{ 0, CV_PI };
            // Offset x = m*x +b --- b = offset
            std::vector<double> offset{ 0.2, 0.4 };
            std::pair<double, double> image_boundaries{ -0.8, 0.8 }; // Boundaries are actually at [-1, 1]. These are boudnareis for lines

            bool valid() const {
                if (points_on_line < 3) return false;
                if (std::any_of(angles.begin(), angles.end(), [](const auto& val) -> bool {
                    return val > CV_PI || val < 0;
                    })) return false;
                if (std::any_of(offset.begin(), offset.end(), [](const auto& val) -> bool {
                    return std::abs(static_cast<double>(val)) > 1.0;
                    })) return false;
                if (std::abs(image_boundaries.first) > 1 || std::abs(image_boundaries.second) > 1)
                    return false;
                return true;
            }
        } lineParameters{};

        [[nodiscard]] bool valid() const {
            if (!lineParameters.valid()) return false;
            if (std::abs(stddev_noise) > 1.0 ) return false;

            return std::isfinite(k1) && std::isfinite(k2) &&
                std::isfinite(k3) && std::isfinite(p1) && std::isfinite(p2);
        }

        bool save_to_file(
            const std::filesystem::path& path,
            const DistortionCalibrationResult& results,
            bool save_point_corrections) const
        {
            std::filesystem::path save_path = path;
            if (!save_path.has_extension())
                save_path.append(".csv");

            // Ensure parent directory exists
            if (save_path.parent_path().empty() && !std::filesystem::create_directories(save_path.parent_path())) {
                std::cout << "Creating files failed: Abort " << std::endl;
                return false;
            }

            bool file_exists = std::filesystem::exists(save_path);

            try {
                std::ofstream file(save_path, std::ios::out | std::ios::app);
                if (!file.is_open()) return false;

                // Write header only if file is newly created
                if (!file_exists) {
                    file << "timestamp,"
                        << "gt_k1,gt_k2,gt_k3,gt_p1,gt_p2,"
                        << "points_per_line,noise_std_dev,"
                        << "res_k1,res_k2,res_k3,res_p1,res_p2\n";
                }

                // Get current timestamp
                std::time_t now = std::time(nullptr);
                char time_buf[32];
                std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

                // Write summary row
                file << time_buf << ","
                    << k1 << "," << k2 << "," << k3 << "," << p1 << "," << p2 << ","
                    << lineParameters.points_on_line << ","
                    << stddev_noise << ","
                    << results.k1 << "," << results.k2 << "," << results.k3 << ","
                    << results.p1 << "," << results.p2 << "\n";

                // If point corrections need to be saved, write them to a separate companion file
                if (save_point_corrections && !results.point_corrections.empty()) {
                    std::filesystem::path pc_path = save_path;
                    pc_path.replace_extension("");
                    std::string pc_filename = pc_path.string() + "_point_corrections.csv";

                    std::ofstream pc_file(pc_filename, std::ios::out);
                    if (pc_file.is_open()) {
                        pc_file << "x,y\n";
                        for (const auto& pt : results.point_corrections) {
                            pc_file << pt.x << "," << pt.y << "\n";
                        }
                    }
                }

                return true;
            }
            catch (const std::exception& e) {
                std::cout << e.what() << std::endl;
                return false;
            }
        }

    };

    static void _TestDistortionCalibration(
        const _TestDistortionCalibrationConfig& config_test,
        const DistortionCalibrationConfig& config,
        const std::filesystem::path& path);

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
