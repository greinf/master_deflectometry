#define VERSION "1"
#include <cstddef>
#include <iostream>
#include <ImageStore.hpp>
#include <open3d/Open3D.h>
#include "Camera.hpp"
#include "Mesh.hpp"
#include "Light.hpp"
#include "Scene.hpp"
#include <opencv2/core/eigen.hpp> // Must include this header
#include "Utils.hpp"


auto showNormalized2Channel = [](const cv::Mat& img, const std::string& winName = "Roflcopter")
    {
        CV_Assert(img.type() == CV_64FC2);

        std::vector<cv::Mat> channels(2);
        cv::split(img, channels);   // -> channels[0] = X, channels[1] = Y

        cv::Mat normX, normY;
        cv::normalize(channels[0], normX, 0, 255, cv::NORM_MINMAX);
        cv::normalize(channels[1], normY, 0, 255, cv::NORM_MINMAX);

        // Optional 3rd channel = zero for visualization
        cv::Mat zero = cv::Mat::zeros(img.rows, img.cols, CV_64F);

        cv::Mat merged;
        cv::merge(std::vector<cv::Mat>{normX, normY, zero}, merged); // 3-Channel Float

        cv::Mat display8u;
        merged.convertTo(display8u, CV_8UC3);

        cv::imshow(winName, display8u);
        cv::waitKey(0);
    };

auto shownormalized = [](const cv::Mat& d) {
    cv::Mat img;
    cv::normalize(d, img, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Erlaubt das freie Skalieren des Fensters per Maus
    cv::namedWindow("norm", cv::WINDOW_NORMAL);

    cv::imshow("norm", img);
    cv::waitKey(0);
    };


int main() 
{
    // Suppress INFO messages, only show WARNING and ERROR
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    const std::string& path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-05-25_CameraCalibration_useGuessIntrinsic/Stereo_Calib.xml" };
    ImageStore img_store{};
    img_store.loadRoleXML(FrameRole::CalibrationMatrix, path);

    std::vector<cv::Mat> camMat{ img_store.get(FrameRole::CalibrationMatrix) };
    std::vector<cv::Mat> distCoeffs{ img_store.get(FrameRole::DistortionCoeff) };

    Scene raycasting{};

    camMat[0].at<double>(0, 2) = 2464.0 / 2;
    camMat[0].at<double>(1, 2) = 2056.0 / 2;
    distCoeffs[0] = cv::Mat(distCoeffs[0].size(), CV_64F, cv::Scalar(0.0));

    double image_scale{ (2464.0 * 3.45e-3 / 1.75 ) / (30 * 12) };

    std::filesystem::path d{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/SyntheticCalibration/Test" };

    Protocoll::SimulationProtocoll protocoll{};

    // protocoll.read(d.string());

    Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Identity();

    Eigen::VectorXd distortion{};

    Eigen::MatrixXd result = Eigen::MatrixXd::Random(2464, 2056);

    distortion.resize(7, 1);

    distortion.fill(1.0);

    protocoll.m_camera.objective = &Protocoll::CameraData::C2514_M;

    protocoll.m_camera.distortionCoefficients = distortion;

    protocoll.m_camera.intrinsicMatrix = intrinsic;

    protocoll.m_calibration.pattern_width = 30;

    protocoll.m_calibration.pattern_x = 9;
    
    protocoll.m_calibration.pattern_y = 12;

    protocoll.m_calibration.translationMatrix = { Eigen::Matrix4d::Random().eval() };

    protocoll.m_simulation.m_simulated_images = { result };

    protocoll.save(d.string());

    std::cout << "Make Lookie lookie \n";

    /*Eigen::Matrix3d intrsinisc = CameraMatrix::generateIntrinsicMatrix(
        &syntheticCalibration::C2514_M,
        16.0,
        image_scale,
        3.45e-3,
        2464 / 2.0,
        2056 / 2.0
    );*/

    SamplingSetting sample{ 6, 6 };

    std::unique_ptr<Camera> cam_ptr = 
        std::make_unique<Camera>(std::make_unique<OpenCvMatrix>(camMat[0], distCoeffs[0]),
            sample);
    
    raycasting.addCamera(std::move(cam_ptr));

    std::unique_ptr<TriangularMesh> parabolical_mirror{ PolygonMesh::ParabolicalMirror(
        1600.0,
        200.0,
        10)->convert2Triangular() };

    Eigen::Matrix4d mirror_trans =
        (Eigen::Matrix4d() << -1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -1, 3700,
            0, 0, 0, 1).finished();

    /* Eigen::Matrix4d mirror_trans1 =
        (Eigen::Matrix4d() <<
            std::cos(CV_PI * 0.99), 0.0, std::sin(CV_PI * 0.99), 0.0,
            0.0, 1.0, 0.0, 0.0,
            -std::sin(CV_PI * 0.99), 0.0, std::cos(CV_PI * 0.99), 4000.0
            ).finished(); */

    parabolical_mirror->addTransform(mirror_trans);

    // raycasting.addObject(std::move(parabolical_mirror));

    std::unique_ptr<TriangularMesh> tubus{ PolygonMesh::TelsecopeTubus(
        1100,
        500,
        0.8,
        0.1,
        10)->convert2Triangular() };

    // tubus->addTransform(tubus_trans);

    // raycasting.addObject(std::move(tubus));

    std::unique_ptr<TriangularMesh> board{ TriangularMesh::CalibrationBoard(
        12,
        9,
        30,
        0.6,
        0.04,
        0.35,
        50)
    };

    Eigen::Matrix4d board_trans =
            (Eigen::Matrix4d() <<
                std::cos(CV_PI * 0.7), 0.0, std::sin(CV_PI * 0.7), 0.0,
                0.0, 1.0, 0.0, 0.0,
                -std::sin(CV_PI * 0.7), 0.0, std::cos(CV_PI * 0.7), 4000.0,
                0, 0, 0, 1
                ).finished();

    board->addTransform(board_trans);

    raycasting.addObject(std::move(board));

    constexpr double width = 1920.0 * 0.2745;
    constexpr double height = 1080.0 * 0.2745;

    const double z = 0.0;

    Eigen::Matrix4d display_trans =
        (Eigen::Matrix4d() <<
            1, 0, 0, -width / 2.0,
            0, 1, 0, - height / 2.0,
            0, 0, 1, z,
            0, 0, 0, 1
            ).finished();

    std::unique_ptr<TriangularMesh> displayMesh{ PolygonMesh::Display()->convert2Triangular() };

    displayMesh->addTransform(display_trans);

    std::unique_ptr<Display> display{ new Display(
        std::move(displayMesh),
        1920,
        1080,
        0.2745,
        108*0.2745,
        4
    ) };

    display->m_info.m_power = 220.0;

    raycasting.addLight(std::move(display));

    std::vector<Eigen::MatrixXd> outputImages{};

    raycasting.raytraceScene(outputImages);

    std::cout << outputImages.size() << '\n';

    cv::Mat output;
    for (const auto& img : outputImages) {
        cv::eigen2cv(outputImages.at(0), output);

        cv::Mat scaledOutput;

        cv::resize(output, scaledOutput, cv::Size(2464, 2056));

        double maxVal;
        // Pass cv::noArray() or NULL if you don't need the minimum value or locations
        cv::minMaxLoc(scaledOutput, nullptr, &maxVal);

        std::cout << "Max coefficient: " << maxVal << std::endl;

        shownormalized(scaledOutput);
    }

    return 0;

}


/*
    open3d::geometry::TriangleMesh mesh(*tubus);

    auto tubus_mesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>(mesh);

    open3d::visualization::DrawGeometries({ tubus_mesh_ptr }, "Custom Mesh Window");
*/