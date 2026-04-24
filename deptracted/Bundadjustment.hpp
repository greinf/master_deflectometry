#ifndef BUNDLEADJUSTMENT_HPP
#define BUNDLEADJUSTMENT_HPP
#include <vector>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

struct PointToRayResidual {
    PointToRayResidual(const Eigen::Vector3d& X_obj,
        const Eigen::Vector3d& O_cam,
        const Eigen::Vector3d& d_cam_unit)
        : Xo(X_obj), O(O_cam), d(d_cam_unit) {
    }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // pose = [angle_axis(3), t(3)]
        // Transform point: p = R * Xo + t  (object -> camera)
        T X[3] = { T(Xo.x()), T(Xo.y()), T(Xo.z()) };
        T p[3];
        ceres::AngleAxisRotatePoint(pose, X, p);
        p[0] += pose[3];
        p[1] += pose[4];
        p[2] += pose[5];

        // v = p - O  (vector from ray origin to point)
        T v0 = p[0] - T(O.x());
        T v1 = p[1] - T(O.y());
        T v2 = p[2] - T(O.z());

        // Remove component along ray direction: r = v - (v·d) d
        // This is the shortest vector from point to the ray's infinite line.
        T dot = v0 * T(d.x()) + v1 * T(d.y()) + v2 * T(d.z());

        residuals[0] = v0 - dot * T(d.x());
        residuals[1] = v1 - dot * T(d.y());
        residuals[2] = v2 - dot * T(d.z());
        return true;
    }

    Eigen::Vector3d Xo; // object point
    Eigen::Vector3d O;  // ray origin (camera coords)
    Eigen::Vector3d d;  // ray direction (camera coords), must be unit
};

inline bool estimatePoseFromRays_Ceres(
    const std::vector<cv::Vec3d>& objectPoints_obj,   // X_obj (your image/object coords in R^3)
    const std::vector<cv::Vec3d>& rayOrigins_cam,     // O_i (hitpoints on mirror, in camera coords)
    const std::vector<cv::Vec3d>& rayDirs_cam,        // d_i (reflected directions, in camera coords)
    cv::Mat& rvec_out,                                  // 3x1 Rodrigues (object->camera)
    cv::Mat& tvec_out,                                  // 3x1
    const cv::Vec3d& rvec_init = cv::Vec3d(0, 0, 0),
    const cv::Vec3d& tvec_init = cv::Vec3d(0, 0, 1.0),     
    bool useRobustLoss = true,
    double huberDelta = 1e-3,                            
    int maxIterations = 80
) {
    const size_t N = objectPoints_obj.size();
    if (N < 6) throw std::runtime_error("Need at least ~6 correspondences for a stable pose-from-rays fit.");
    if (rayOrigins_cam.size() != N || rayDirs_cam.size() != N)
        throw std::runtime_error("Input vectors must have the same length.");

    // pose: angle-axis (3) + translation (3)
    double pose[6] = {
        rvec_init[0], rvec_init[1], rvec_init[2],
        tvec_init[0], tvec_init[1], tvec_init[2]
    };

    ceres::Problem problem;

    ceres::LossFunction* loss = useRobustLoss ? new ceres::HuberLoss(huberDelta) : nullptr;

    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xo(objectPoints_obj[i][0], objectPoints_obj[i][1], objectPoints_obj[i][2]);
        Eigen::Vector3d O(rayOrigins_cam[i][0], rayOrigins_cam[i][1], rayOrigins_cam[i][2]);

        Eigen::Vector3d d(rayDirs_cam[i][0], rayDirs_cam[i][1], rayDirs_cam[i][2]);
        const double n = d.norm();
        if (n <= 0.0) throw std::runtime_error("Ray direction with zero length encountered.");
        d /= n; // normalize

        auto* cost =
            new ceres::AutoDiffCostFunction<PointToRayResidual, 3, 6>(
                new PointToRayResidual(Xo, O, d));

        problem.AddResidualBlock(cost, loss, pose);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = maxIterations;
    options.linear_solver_type = ceres::DENSE_QR;   // 6 params -> dense is fine
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Output as OpenCV Mats
    rvec_out = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);
    tvec_out = (cv::Mat_<double>(3, 1) << pose[3], pose[4], pose[5]);

    return summary.IsSolutionUsable();
}


#endif