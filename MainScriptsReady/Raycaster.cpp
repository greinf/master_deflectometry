#define VERSION "1"
#include <cstddef>
#include <iostream>
#include "deflectometry.hpp"
#include "enums.hpp"
#include <opencv2/opencv.hpp>
#include "acquisitionworker.hpp"
#include <filesystem>
#include <regex>
#include "GrayCodeDecoder.hpp"
#include "screen.hpp"
#include "imgProcessing.hpp"
#include "CameraSimulation.hpp"
// #include "Mesh.hpp"
#include <open3d/Open3D.h>
#include "imageStore.hpp"
#include "Camera.hpp"
#include "Scene.hpp"


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
    cv::imshow("norm", img);
    cv::waitKey(0);
    };


int main()
{
    const std::string& path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-05-25_CameraCalibration_useGuessIntrinsic/Stereo_Calib.xml" };
    ImageStore img_store{};
    img_store.loadRoleXML(FrameRole::CalibrationMatrix, path);

    std::vector<cv::Mat> camMat{ img_store.get(FrameRole::CalibrationMatrix) };
    std::vector<cv::Mat> distCoeffs{ img_store.get(FrameRole::DistortionCoeff) };

    Scene raycasting{};

    std::unique_ptr<Camera> cam_ptr =
        std::make_unique<Camera>(std::make_unique<OpenCvMatrix>(camMat[0], distCoeffs[0]));

    raycasting.addCamera(std::move(cam_ptr));

    std::vector<Eigen::MatrixXd> outputImages{};

    raycasting.raytraceScene(outputImages);

    return 0;
    std::cout << "dodod \n";


    //// std::unique_ptr<PolygonMesh> mesh{std::make_unique<PolygonMesh>(PolygonMesh::Pa)
    //std::unique_ptr<PolygonMesh> mesh{ std::move(PolygonMesh::ParabolicalMirror(
    //    1600.0,
    //    200.0,
    //    500)) };
    //
    //auto triangle{ mesh->convert2Triangular() };

    //auto open3dmesh = static_cast<open3d::geometry::TriangleMesh>(*triangle);

    //open3dmesh.PaintUniformColor({ 0.7, 0.7, 0.7 });

    //auto open3dmesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>(
    //    std::move(open3dmesh)
    //);

    //std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries{};
    //geometries.push_back(open3dmesh_ptr);

    //open3d::visualization::DrawGeometries(
    //    geometries,
    //    "Mesh visualization",
    //    1280,
    //    720
    //);


    return 0;
}



