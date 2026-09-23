#define VERSION "1"
#include <cstddef>
#include <iostream>
#include <ImageStore.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp> // Must include this header
#include "Scene.hpp"
#include "Utils.hpp"
#include "SyntheticCalibration.hpp"
#include "CalibrationCallAble.hpp"

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
    

    double image_scale{ (2464.0 * 3.45e-3 / 1.75 ) / (30.0 * 12) };

    SamplingSetting sample{ 5, 5 };

    std::filesystem::path d{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/SyntheticCalibration/" };
    
    SyntheticCalibration raycastingCalibration{};

    std::vector<const Protocoll::CameraData::ObjectiveData*> objectives{ {
            //&Protocoll::CameraData::C2514_M,
            //&Protocoll::CameraData::C3516_M
            &Protocoll::CameraData::C5028_M
            //&Protocoll::CameraData::C7528_M
            }
    };

    Eigen::VectorXd dist = (Eigen::VectorXd(5) <<
        -1.0, 2.0, 0.0, 0.0, 0.0).finished();
    
    raycastingCalibration.build(
        image_scale,
        objectives,
        {},
        15,
        sample
    );

    std::vector<Protocoll::SimulationProtocoll> protocoll = raycastingCalibration.calibrations();

    Protocoll::BatchIO::renderAll(protocoll, Tracing_CallAble);

    Protocoll::BatchIO::saveAll(protocoll, d);

    protocoll[0].m_simulation.showAll();

    //std::vector<Protocoll::SimulationProtocoll> protocoll_{ Protocoll::BatchIO::loadAll(d) };

    //if (protocoll_.size() != protocoll.size()) {
    //    std::cout << "wait\n";
    //}

    //for (std::size_t i = 0; i < protocoll_.size(); ++i) {
    //    if (!(protocoll[i] == protocoll_[i]))
    //        std::cout << i << "Failed \n";
    //}
   
    //// auto protocol = Protocoll::BatchIO::loadAll(d);

    //// protocoll[0].m_simulation.showAll();

    //// protocoll[1].m_simulation.showAll();

    //std::cout << "Readout \n";
    
    //std::unique_ptr<TriangularMesh> parabolical_mirror{ PolygonMesh::ParabolicalMirror(
    //    1600.0,
    //    200.0,
    //    10)->convert2Triangular() };

    //Eigen::Matrix4d mirror_trans =
    //    (Eigen::Matrix4d() << -1, 0, 0, 0,
    //        0, 1, 0, 0,
    //        0, 0, -1, 3700,
    //        0, 0, 0, 1).finished();

    ///* Eigen::Matrix4d mirror_trans1 =
    //    (Eigen::Matrix4d() <<
    //        std::cos(CV_PI * 0.99), 0.0, std::sin(CV_PI * 0.99), 0.0,
    //        0.0, 1.0, 0.0, 0.0,
    //        -std::sin(CV_PI * 0.99), 0.0, std::cos(CV_PI * 0.99), 4000.0
    //        ).finished(); */

    //parabolical_mirror->addTransform(mirror_trans);

    //// raycasting.addObject(std::move(parabolical_mirror));

    //std::unique_ptr<TriangularMesh> tubus{ PolygonMesh::TelsecopeTubus(
    //    1100,
    //    500,
    //    0.8,
    //    0.1,
    //    10)->convert2Triangular() };

    // tubus->addTransform(tubus_trans);

    // raycasting.addObject(std::move(tubus));

    /*Eigen::Matrix4d board_trans =
            (Eigen::Matrix4d() <<
                std::cos(CV_PI * 0.7), 0.0, std::sin(CV_PI * 0.7), 0.0,
                0.0, 1.0, 0.0, 0.0,
                -std::sin(CV_PI * 0.7), 0.0, std::cos(CV_PI * 0.7), 4000.0,
                0, 0, 0, 1
                ).finished();

    board->addTransform(board_trans);

    raycasting.addObject(std::move(board));*/

    //constexpr double width = 1920.0 * 0.2745;
    //constexpr double height = 1080.0 * 0.2745;

    //const double z = 0.0;

    //Eigen::Matrix4d display_trans =
    //    (Eigen::Matrix4d() <<
    //        1, 0, 0, -width / 2.0,
    //        0, 1, 0, - height / 2.0,
    //        0, 0, 1, z,
    //        0, 0, 0, 1
    //        ).finished();

    //std::unique_ptr<TriangularMesh> displayMesh{ PolygonMesh::Display()->convert2Triangular() };

    //displayMesh->addTransform(display_trans);

    //std::unique_ptr<Display> display{ new Display(
    //    std::move(displayMesh),
    //    1920,
    //    1080,
    //    0.2745,
    //    108*0.2745,
    //    4
    //) };

    //display->m_info.m_power = 220.0;

    //raycasting.addLight(std::move(display));

    return 0;
}


/*
    open3d::geometry::TriangleMesh mesh(*tubus);

    auto tubus_mesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>(mesh);

    open3d::visualization::DrawGeometries({ tubus_mesh_ptr }, "Custom Mesh Window");
*/

/*img_store.loadRoleXML(FrameRole::CalibrationMatrix, path);

    std::vector<cv::Mat> camMat{ img_store.get(FrameRole::CalibrationMatrix) };
    std::vector<cv::Mat> distCoeffs{ img_store.get(FrameRole::DistortionCoeff) };


    camMat[0].at<double>(0, 2) = 2464.0 / 2;
    camMat[0].at<double>(1, 2) = 2056.0 / 2;
    distCoeffs[0] = cv::Mat(distCoeffs[0].size(), CV_64F, cv::Scalar(0.0));*/