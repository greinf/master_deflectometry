#ifndef SYNTHETICCALIBRATION_HPP
#define SYNTHETICCALIBRATION_HPP

#include "Utils.hpp"
#include "ProtocollBatchIO.hpp" 
#include "CameraCalibration.hpp"
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <random>
#include <Eigen/dense>
#include <opencv2/opencv.hpp>
#include <algorithm>


class SyntheticCalibration {
public:

    struct PoseMatchQuality {
        double rms_px{};
        double max_px{};
    };

    static void doSyntheticCalibration(
        const std::vector<Protocoll::SimulationProtocoll>& protocoll)
    {
        if (protocoll.empty()) throw std::invalid_argument("Protocoll contains no Data");
        if (std::all_of(protocoll.begin(), protocoll.end(), 
            [](const Protocoll::SimulationProtocoll& protocoll) -> bool {
                return protocoll.validAndSet();
            })) throw std::invalid_argument("Protocoll data is not fully set");


    }

    SyntheticCalibration() = default;

    //static do_Calibration() 

    void build(
        const double image_magnification,
        const std::vector<const Protocoll::CameraData::ObjectiveData*>& available_objectives,
        const std::vector<Eigen::VectorXd>& distortion_coefficients,
        const std::size_t n_translations,
        const SamplingSetting& settings,
        const double board_rotation_limit = CV_PI / 4.0,
        const std::pair<int, int> calibBoardDimension = { 12, 9 }, // (x,y)
        const double calibPattern_width = 30.0,
        const double f_number = 16.0,
        const std::pair<int, int>& sensor_dimension = { 2464, 2056 },
        const double pixel_pitch = 3.45e-3,
        const double image_border_margin_px = 20.0,
        // Optional z variation of the REFERENCE board center [mm].
        // {0,0} means: no z variation.
        const std::pair<double, double>& reference_z_offset_range_mm = { 0.0, 0.0 },
        // Dense support grid used to match the image warp across the board.
        const std::pair<int, int>& matching_support_grid = { 11, 15 },
        const std::uint64_t random_seed = 42ULL)
    {
        validateInputs(
            image_magnification,
            available_objectives,
            n_translations,
            board_rotation_limit,
            calibBoardDimension,
            calibPattern_width,
            sensor_dimension,
            pixel_pitch,
            image_border_margin_px,
            reference_z_offset_range_mm,
            matching_support_grid);

        m_calibrations.clear();
        m_match_quality.clear();

        const auto distortions = buildDistortionVariations(distortion_coefficients);

        // Build every camera setup first.
        for (const auto* objective : available_objectives) {
            if (objective == nullptr)
                throw std::invalid_argument("Null objective pointer supplied");

            if (!objective->valid_Iris(f_number)) {
                throw std::invalid_argument(
                    "Requested f-number is unavailable for objective " + objective->name);
            }

            const Eigen::Matrix3d intrinsic = generateIntrinsicMatrix(
                objective,
                image_magnification,
                pixel_pitch,
                sensor_dimension.first / 2.0 - 0.5,
                sensor_dimension.second / 2.0 - 0.5);

            const double object_length =
                (image_magnification + 1.0) * objective->focal_length /
                image_magnification;

            const auto focus_points = objective->generateFocusPoints(
                image_magnification,
                f_number,
                pixel_pitch,
                object_length);

            if (focus_points.FarPoint <= object_length ||
                focus_points.NearPoint >= object_length)
            {
                throw std::invalid_argument("Nominal object lies outside the focus area");
            }

            for (const auto& distortion : distortions) {
                Protocoll::SimulationProtocoll setup{};

                setup.m_camera.intrinsicMatrix = intrinsic;
                setup.m_camera.distortionCoefficients = distortion;
                setup.m_camera.focus_points = focus_points;
                setup.m_camera.used_f_num = f_number;
                setup.m_camera.objective = objective;

                setup.m_camera.focus_points.FarPoint = focus_points.FarPoint;
                setup.m_camera.focus_points.NearPoint = focus_points.NearPoint;
                setup.m_camera.focus_points.ObjectLength = focus_points.ObjectLength;

                setup.m_simulation.settings = settings;

                setup.m_calibration.pattern_x =
                    checkedCastToUint16(calibBoardDimension.first, "pattern_x");
                setup.m_calibration.pattern_y =
                    checkedCastToUint16(calibBoardDimension.second, "pattern_y");
                setup.m_calibration.pattern_width = calibPattern_width;

                m_calibrations.push_back(std::move(setup));
            }
        }

        if (m_calibrations.empty())
            throw std::runtime_error("No simulation protocols were generated");

        // Reference geometry in board-local coordinates.
        // Native Coordinate System is in the upper left Corner.
        // For the following Calculation we Transfer the coordiante system in the board center
        const auto board_corners = makeBoardCorners(
            calibBoardDimension,
            calibPattern_width);

        const Eigen::Vector3d board_local_center = boardCenter(
            calibBoardDimension,
            calibPattern_width);

        // For solvePnP a few more support points are needed. 
        const std::vector<cv::Point3d> support_points = makeBoardSupportPoints(
            calibBoardDimension,
            calibPattern_width,
            matching_support_grid);

        // First camera setup is deliberately first objective + zero distortion.
        const auto& reference_setup = m_calibrations.front();
        const double reference_nominal_z =
            reference_setup.m_camera.focus_points.ObjectLength;

        std::mt19937_64 generator{ random_seed };

        // One quality entry per simulation setup, one value per pose.
        m_match_quality.resize(m_calibrations.size());
        for (auto& q : m_match_quality)
            q.reserve(n_translations);

        for (std::size_t pose_index = 0; pose_index < n_translations; ++pose_index) {

            // Create ONE valid random pose in the REFERENCE camera.
            const Eigen::Matrix4d reference_pose = generateReferencePose(
                reference_setup.m_camera.intrinsicMatrix,
                reference_setup.m_camera.distortionCoefficients,
                reference_nominal_z,
                reference_setup.m_camera.focus_points,
                board_rotation_limit,
                board_corners,
                board_local_center,
                sensor_dimension,
                image_border_margin_px,
                reference_z_offset_range_mm,
                generator);

            // Reference distorted pixel coordinates become the common target.
            const std::vector<cv::Point2d> target_pixels = projectPoints(
                support_points,
                reference_pose,
                reference_setup.m_camera.intrinsicMatrix,
                reference_setup.m_camera.distortionCoefficients);

            // The reference setup gets its own pose unchanged.
            m_calibrations.front().m_calibration.translationMatrix.push_back(reference_pose);
            m_match_quality.front().push_back(PoseMatchQuality{ 0.0, 0.0 });

            // -------------------------------------------------------------
            // 4) For every other K / distortion model, find a board pose
            //    whose DISTORTED projection best matches target_pixels.
            // -------------------------------------------------------------
            for (std::size_t setup_index = 1;
                setup_index < m_calibrations.size();
                ++setup_index)
            {
                auto& target_setup = m_calibrations[setup_index];

                const double target_nominal_z =
                    target_setup.m_camera.focus_points.ObjectLength;

                const Eigen::Matrix4d initial_guess = makeScaledInitialGuess(
                    reference_pose,
                    board_local_center,
                    reference_nominal_z,
                    target_nominal_z);

                const ImageMatchedPose result = solveImageMatchedPose(
                    support_points,
                    target_pixels,
                    target_setup.m_camera.intrinsicMatrix,
                    target_setup.m_camera.distortionCoefficients,
                    initial_guess,
                    sensor_dimension,
                    image_border_margin_px);

                target_setup.m_calibration.translationMatrix.push_back(result.pose);
                m_match_quality[setup_index].push_back(result.quality);
            }
        }
    }

    const std::vector<Protocoll::SimulationProtocoll>& calibrations() const noexcept {
        return m_calibrations;
    }

    const std::vector<std::vector<PoseMatchQuality>>& matchQuality() const noexcept {
        return m_match_quality;
    }

private:
    struct ImageMatchedPose {
        Eigen::Matrix4d pose{ Eigen::Matrix4d::Identity() };
        PoseMatchQuality quality{};
    };

    static void validateInputs(
        const double image_magnification,
        const std::vector<const Protocoll::CameraData::ObjectiveData*>& available_objectives,
        const std::size_t n_translations,
        const double board_rotation_limit,
        const std::pair<int, int>& board_dimension,
        const double pattern_width,
        const std::pair<int, int>& sensor_dimension,
        const double pixel_pitch,
        const double margin_px,
        const std::pair<double, double>& z_offset_range,
        const std::pair<int, int>& support_grid)
    {
        if (n_translations == 0)
            throw std::invalid_argument("n_translations must be bigger than zero");
        if (image_magnification <= 0.0 || image_magnification > 1.0)
            throw std::invalid_argument("image_magnification must be in (0, 1]");
        if (available_objectives.empty())
            throw std::invalid_argument("No objectives supplied");
        if (sensor_dimension.first <= 0 || sensor_dimension.second <= 0)
            throw std::invalid_argument("Sensor dimensions must be bigger than zero");
        if (pixel_pitch <= 0.0)
            throw std::invalid_argument("Pixel pitch must be bigger than zero");
        if (board_dimension.first <= 0 || board_dimension.second <= 0)
            throw std::invalid_argument("Board dimensions must be bigger than zero");
        if (pattern_width <= 0.0)
            throw std::invalid_argument("Pattern width must be bigger than zero");
        if (board_rotation_limit < 0.0 || board_rotation_limit >= CV_PI / 2.0)
            throw std::invalid_argument("board_rotation_limit must be in [0, pi/2)");
        if (margin_px < 0.0 ||
            2.0 * margin_px >= static_cast<double>(sensor_dimension.first) ||
            2.0 * margin_px >= static_cast<double>(sensor_dimension.second))
        {
            throw std::invalid_argument("Invalid image border margin");
        }
        if (z_offset_range.first > z_offset_range.second)
            throw std::invalid_argument("reference_z_offset_range_mm is reversed");
        if (support_grid.first < 2 || support_grid.second < 2)
            throw std::invalid_argument("matching_support_grid must be at least 2x2");
    }

    static std::vector<Eigen::VectorXd> buildDistortionVariations(
        const std::vector<Eigen::VectorXd>& distortion_coefficients)
    {
        std::vector<Eigen::VectorXd> result{};
        result.emplace_back(Eigen::VectorXd::Zero(5));

        for (const auto& distortion : distortion_coefficients) {
            if (distortion.size() == 0 || distortion.isZero())
                continue;

            result.push_back(distortion);
        }

        std::cout << result.size() << " distortion variations will be tested\n"
            << "First variation is always zero distortion\n";
        return result;
    }

    static std::uint16_t checkedCastToUint16(const int value, const char* name) {
        if (value < 0 ||
            value > static_cast<int>(std::numeric_limits<std::uint16_t>::max()))
        {
            throw std::invalid_argument(std::string(name) + " does not fit into uint16_t");
        }
        return static_cast<std::uint16_t>(value);
    }

    static double boardWidth(
        const std::pair<int, int>& board_dimension,
        const double pattern_width)
    {
        return static_cast<double>(board_dimension.first) * pattern_width;
    }

    static double boardHeight(
        const std::pair<int, int>& board_dimension,
        const double pattern_width)
    {
        return static_cast<double>(board_dimension.second) * pattern_width;
    }

    static std::array<Eigen::Vector3d, 4> makeBoardCorners(
        const std::pair<int, int>& board_dimension,
        const double pattern_width)
    {
        const double width = boardWidth(board_dimension, pattern_width);
        const double height = boardHeight(board_dimension, pattern_width);

        return {
            Eigen::Vector3d{0.0,   0.0,    0.0},
            Eigen::Vector3d{width, 0.0,    0.0},
            Eigen::Vector3d{0.0,   height, 0.0},
            Eigen::Vector3d{width, height, 0.0}
        };
    }

    static Eigen::Vector3d boardCenter(
        const std::pair<int, int>& board_dimension,
        const double pattern_width)
    {
        return {
            0.5 * boardWidth(board_dimension, pattern_width),
            0.5 * boardHeight(board_dimension, pattern_width),
            0.0
        };
    }

    static std::vector<cv::Point3d> makeBoardSupportPoints(
        const std::pair<int, int>& board_dimension,
        const double pattern_width,
        const std::pair<int, int>& support_grid)
    {
        const double width = boardWidth(board_dimension, pattern_width);
        const double height = boardHeight(board_dimension, pattern_width);

        std::vector<cv::Point3d> points{};
        points.reserve(static_cast<std::size_t>(
            support_grid.first * support_grid.second));

        for (int iy = 0; iy < support_grid.second; ++iy) {
            const double y = height * static_cast<double>(iy) /
                static_cast<double>(support_grid.second - 1);

            for (int ix = 0; ix < support_grid.first; ++ix) {
                const double x = width * static_cast<double>(ix) /
                    static_cast<double>(support_grid.first - 1);
                points.emplace_back(x, y, 0.0);
            }
        }
        return points;
    }

    static cv::Mat eigenIntrinsicToCv(const Eigen::Matrix3d& K) {
        cv::Mat result(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                result.at<double>(r, c) = K(r, c);
        return result;
    }

    static cv::Mat eigenDistortionToCv(const Eigen::VectorXd& d) {
        if (d.size() == 0)
            return cv::Mat{};

        cv::Mat result(1, static_cast<int>(d.size()), CV_64F);
        for (Eigen::Index i = 0; i < d.size(); ++i)
            result.at<double>(0, static_cast<int>(i)) = d(i);
        return result;
    }

    static Eigen::Matrix3d randomBoundedRotation(
        const double rotation_limit,
        std::mt19937_64& generator)
    {
        std::uniform_real_distribution<double> dist_xy(
            -rotation_limit,
            rotation_limit);
        std::uniform_real_distribution<double> dist_z(0.0, CV_2PI);

        const double rx = dist_xy(generator);
        const double ry = dist_xy(generator);
        const double rz = dist_z(generator);

        return (
            Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX())
            ).toRotationMatrix();
    }

    static Eigen::Matrix4d makePoseFromBoardCenter(
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& local_board_center,
        const Eigen::Vector3d& board_center_camera)
    {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = rotation;
        pose.block<3, 1>(0, 3) =
            board_center_camera - rotation * local_board_center;
        return pose;
    }

    static std::vector<cv::Point2d> projectPoints(
        const std::vector<cv::Point3d>& object_points,
        const Eigen::Matrix4d& pose,
        const Eigen::Matrix3d& intrinsic,
        const Eigen::VectorXd& distortion)
    {
        const cv::Mat K = eigenIntrinsicToCv(intrinsic);
        const cv::Mat D = eigenDistortionToCv(distortion);

        cv::Mat R(3, 3, CV_64F);
        cv::Mat tvec(3, 1, CV_64F);

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                R.at<double>(r, c) = pose(r, c);
            }
            tvec.at<double>(r, 0) = pose(r, 3);
        }

        cv::Mat rvec{};
        cv::Rodrigues(R, rvec);

        std::vector<cv::Point2d> image_points{};
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            D,
            image_points);

        return image_points;
    }

    static bool allPixelsInside(
        const std::vector<cv::Point2d>& pixels,
        const std::pair<int, int>& sensor_dimension,
        const double margin_px)
    {
        const double max_u = static_cast<double>(sensor_dimension.first - 1) - margin_px;
        const double max_v = static_cast<double>(sensor_dimension.second - 1) - margin_px;

        for (const auto& p : pixels) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y))
                return false;
            if (p.x < margin_px || p.x > max_u ||
                p.y < margin_px || p.y > max_v)
            {
                return false;
            }
        }
        return true;
    }

    // Method creates Random Rotation Transformation and test if the hole Object is visible
    // The method also test if the hole object is in focus with respect to near and far point -> 
    // circle of confusion with respect to the pixelPitch on the sensor
    static Eigen::Matrix4d generateReferencePose(
        const Eigen::Matrix3d& intrinsic,
        const Eigen::VectorXd& distortion,
        const double nominal_z,
        const Protocoll::CameraData::ObjectiveData::FocusPoints& focus_points,
        const double rotation_limit,
        const std::array<Eigen::Vector3d, 4>& board_corners,
        const Eigen::Vector3d& local_board_center,
        const std::pair<int, int>& sensor_dimension,
        const double margin_px,
        const std::pair<double, double>& z_offset_range,
        std::mt19937_64& generator)
    {
        constexpr std::size_t max_attempts = 20000;

        std::uniform_real_distribution<double> z_offset_dist(
            z_offset_range.first,
            z_offset_range.second);

        // We sample the board-center position in a conservative world-space
        // window and accept/reject using the REAL distorted projection.
        // The window is estimated from image size at the sampled z.
        const double fx = intrinsic(0, 0);
        const double fy = intrinsic(1, 1);
        const double cx = intrinsic(0, 2);
        const double cy = intrinsic(1, 2);

        if (fx <= 0.0 || fy <= 0.0)
            throw std::invalid_argument("Invalid intrinsic focal length");

        std::vector<cv::Point3d> corner_points{};
        corner_points.reserve(4);
        for (const auto& p : board_corners)
            corner_points.emplace_back(p.x(), p.y(), p.z());

        for (std::size_t attempt = 0; attempt < max_attempts; ++attempt) {
            const double center_z = nominal_z + z_offset_dist(generator);

            if (center_z <= 0.0)
                continue;

            // Keep optional z variation inside the nominal depth-of-field.
            if (center_z <= focus_points.NearPoint ||
                center_z >= focus_points.FarPoint)
            {
                continue;
            }

            const Eigen::Matrix3d R = randomBoundedRotation(
                rotation_limit,
                generator);

            // Rough x/y range corresponding to the full image plane at center_z.
            // The final acceptance test below uses the full distortion model.
            const double x_left = (margin_px - cx) / fx * center_z;
            const double x_right =
                (static_cast<double>(sensor_dimension.first - 1) - margin_px - cx) /
                fx * center_z;
            const double y_top = (margin_px - cy) / fy * center_z;
            const double y_bottom =
                (static_cast<double>(sensor_dimension.second - 1) - margin_px - cy) /
                fy * center_z;

            if (x_left >= x_right || y_top >= y_bottom)
                continue;

            std::uniform_real_distribution<double> dist_x(x_left, x_right);
            std::uniform_real_distribution<double> dist_y(y_top, y_bottom);

            const Eigen::Vector3d center_cam{
                dist_x(generator),
                dist_y(generator),
                center_z
            };

            const Eigen::Matrix4d pose = makePoseFromBoardCenter(
                R,
                local_board_center,
                center_cam);

            // Ensure every physical board corner is in front of the camera.
            bool positive_depth = true;
            for (const auto& local_corner : board_corners) {
                const Eigen::Vector4d h{
                    local_corner.x(),
                    local_corner.y(),
                    local_corner.z(),
                    1.0
                };
                const Eigen::Vector4d cam = pose * h;
                if (cam.z() <= 1e-9) {
                    positive_depth = false;
                    break;
                }
            }
            if (!positive_depth)
                continue;

            const auto projected_corners = projectPoints(
                corner_points,
                pose,
                intrinsic,
                distortion);

            if (!allPixelsInside(
                projected_corners,
                sensor_dimension,
                margin_px))
            {
                continue;
            }

            return pose;
        }

        throw std::runtime_error(
            "Could not generate a valid reference calibration-board pose");
    }

    // Assumes the same Rotation while only scaling the translation vector with respect
    // to the assumed image scaling - that is similar for all objectives
    static Eigen::Matrix4d makeScaledInitialGuess(
        const Eigen::Matrix4d& reference_pose,
        const Eigen::Vector3d& local_board_center,
        const double reference_nominal_z,
        const double target_nominal_z)
    {
        if (reference_nominal_z <= 0.0 || target_nominal_z <= 0.0)
            throw std::invalid_argument("Nominal z must be positive");

        // Same Rotations Matrix
        const Eigen::Matrix3d R = reference_pose.block<3, 3>(0, 0);

        // Transformation to coordinate System in the board center
        // It is much easier to test the Focus area if we rotate around the middle of the board
        const Eigen::Vector3d reference_center_cam =
            R * local_board_center +
            reference_pose.block<3, 1>(0, 3);

        const double scale = target_nominal_z / reference_nominal_z;
        const Eigen::Vector3d target_center_guess =
            reference_center_cam * scale;

        // Transformation back to board coordinate System top left
        return makePoseFromBoardCenter(
            R,
            local_board_center,
            target_center_guess);
    }

    static void poseToOpenCv(
        const Eigen::Matrix4d& pose,
        cv::Mat& rvec,
        cv::Mat& tvec)
    {
        cv::Mat R(3, 3, CV_64F);
        tvec = cv::Mat(3, 1, CV_64F);

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c)
                R.at<double>(r, c) = pose(r, c);
            tvec.at<double>(r, 0) = pose(r, 3);
        }

        cv::Rodrigues(R, rvec);
    }

    static Eigen::Matrix4d openCvToPose(
        const cv::Mat& rvec,
        const cv::Mat& tvec)
    {
        cv::Mat R{};
        cv::Rodrigues(rvec, R);

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c)
                pose(r, c) = R.at<double>(r, c);
            pose(r, 3) = tvec.at<double>(r, 0);
        }
        return pose;
    }

    static PoseMatchQuality calculateQuality(
        const std::vector<cv::Point2d>& target_pixels,
        const std::vector<cv::Point2d>& actual_pixels)
    {
        if (target_pixels.size() != actual_pixels.size() || target_pixels.empty())
            throw std::invalid_argument("Pixel vectors have incompatible sizes");

        double sum_squared = 0.0;
        double max_error = 0.0;

        for (std::size_t i = 0; i < target_pixels.size(); ++i) {
            const double dx = actual_pixels[i].x - target_pixels[i].x;
            const double dy = actual_pixels[i].y - target_pixels[i].y;
            const double e = std::sqrt(dx * dx + dy * dy);

            sum_squared += e * e;
            max_error = std::max(max_error, e);
        }

        return PoseMatchQuality{
            std::sqrt(sum_squared / static_cast<double>(target_pixels.size())),
            max_error
        };
    }

    static Eigen::Matrix3d generateIntrinsicMatrix(
        const Protocoll::CameraData::ObjectiveData* objective,
        const double& image_scale_goal_abs,
        const double& pixel_pitch,
        const double& px,
        const double& py,
        std::pair<int, int>& sensor_dimension = std::pair<int, int>(2464, 2056),
        const double& fx_fy_ratio = 1.0 /*fx/fy*/)
    {
        if (fx_fy_ratio != 1.0) std::cout << "WARNING: Ratio fx to fy is != 0 \n";
        if (objective == nullptr) throw std::invalid_argument("Objective Data is nullptr");
        if (image_scale_goal_abs > 1.0 || image_scale_goal_abs <= 0.0) throw
            std::invalid_argument("ImageScale must be a positive value smaller than 1.0");
        if (pixel_pitch <= 0.0) throw std::invalid_argument("Pixel Pitch must greater than zero");


        // Debugging
        /*{
            std::cout << "Object to Hauptebene: " << object_length << '\n';
            std::cout << "Nahpunkt: " << focusPoints.NearPoint << '\n';
            std::cout << "Fernpunkt: " << focusPoints.FarPoint << '\n';
        }*/

        const double corrected_focal_length{ (image_scale_goal_abs + 1) * objective->focal_length };

        const double normalized_focal{
            corrected_focal_length / pixel_pitch
        };

        const double normalized_fx{
            normalized_focal * fx_fy_ratio
        };

        Eigen::Matrix3d intrinsic =
            (Eigen::Matrix3d() <<
                normalized_fx, 0.0, px,
                0.0, normalized_focal, py,
                0.0, 0.0, 1.0).finished();

        return intrinsic;
    }

    // Method tries to find the best available Transformation that places the Calibration Board
    // for each objective in the same sensorcoordiantes after projection
    // Takes an initial gues and than does cv::solvePnP for the given camera objective
    static ImageMatchedPose solveImageMatchedPose(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& target_pixels,
        const Eigen::Matrix3d& target_intrinsic,
        const Eigen::VectorXd& target_distortion,
        const Eigen::Matrix4d& initial_guess,
        const std::pair<int, int>& sensor_dimension,
        const double margin_px)
    {
        if (object_points.size() != target_pixels.size() || object_points.size() < 4)
            throw std::invalid_argument("Need at least four 3D/2D correspondences");

        const cv::Mat K = eigenIntrinsicToCv(target_intrinsic);
        const cv::Mat D = eigenDistortionToCv(target_distortion);

        cv::Mat rvec{}, tvec{};
        poseToOpenCv(initial_guess, rvec, tvec);

        // ITERATIVE performs nonlinear reprojection minimization and accepts
        // an extrinsic initial guess. That is useful here because all support
        // points are coplanar and we want to stay on the physically matching
        // branch near the reference orientation.
        const bool ok = cv::solvePnP(
            object_points,
            target_pixels,
            K,
            D,
            rvec,
            tvec,
            true,
            cv::SOLVEPNP_ITERATIVE);

        if (!ok)
            throw std::runtime_error("solvePnP failed while matching camera image");

        // Explicit nonlinear LM refinement of the same reprojection objective.
        cv::solvePnPRefineLM(
            object_points,
            target_pixels,
            K,
            D,
            rvec,
            tvec);

        const Eigen::Matrix4d pose = openCvToPose(rvec, tvec);

        // Reject a physically invalid planar solution.
        for (const auto& p : object_points) {
            const Eigen::Vector4d h{ p.x, p.y, p.z, 1.0 };
            if ((pose * h).z() <= 1e-9)
                throw std::runtime_error("Image-matched pose lies behind the camera");
        }

        const auto actual_pixels = projectPoints(
            object_points,
            pose,
            target_intrinsic,
            target_distortion);

        if (!allPixelsInside(actual_pixels, sensor_dimension, margin_px)) {
            throw std::runtime_error(
                "Matched pose leaves the requested image area");
        }

        return ImageMatchedPose{
            pose,
            calculateQuality(target_pixels, actual_pixels)
        };
    }

private:
    std::vector<Protocoll::SimulationProtocoll> m_calibrations{};
    std::vector<std::vector<PoseMatchQuality>> m_match_quality{};
};








#endif // SYNTHETICCALIBRATION_HPP	