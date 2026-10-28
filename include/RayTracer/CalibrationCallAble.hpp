#ifndef CALLIBRATIONCALLABLE_HPP	
#define CALLIBRATIONCALLABLE_HPP

#include "Scene.hpp"
#include "Utils.hpp"
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include <iostream>

inline auto Tracing_CallAble = [](
    Protocoll::SimulationProtocoll& protocoll,
    const std::size_t setupIndex,
    const std::size_t poseIndex) -> void
    {
        // Create Objects
        const auto& current = protocoll;
        current.m_simulation.validateMetadata();
        if (!current.m_simulation.centered_subsamples)
            throw std::runtime_error("New rendering requires centered_subsamples=true; keep old images separate");

        Eigen::Matrix4d board_trans =
            (Eigen::Matrix4d() <<
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, -1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
                ).finished();

        std::unique_ptr<TriangularMesh> board{ TriangularMesh::CalibrationBoard(
            current.m_calibration.pattern_x,
            current.m_calibration.pattern_y,
            current.m_calibration.pattern_width)
        };

        auto mat = current.m_calibration.translationMatrix[poseIndex]; // *board_trans;

        std::cout << current.m_calibration.translationMatrix[poseIndex] << std::endl;

        std::cout << mat << std::endl;

        std::cout << current.m_camera.intrinsicMatrix;

        board->addTransform(mat);

        std::unique_ptr<Camera> cam{
            std::make_unique<Camera>(
                std::make_unique<OpenCvMatrix>(current.m_camera.intrinsicMatrix, current.m_camera.distortionCoefficients),
                CameraMatrix::generateSensorCoords(0, 0,
                    current.m_simulation.target_image_size.width, current.m_simulation.target_image_size.height),
                current.m_simulation.settings)
        };

        std::unique_ptr<AreaLight> exampleLight{
            AreaLight::generateSimpleAreaLight(400, 400, 255) };

        Eigen::Matrix4d lightTrans = 
            (Eigen::Matrix4d() <<
                -1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
                ).finished();

        exampleLight->addTransform(lightTrans);

        // Scene Object fill it with objects
        Scene raycasting{};
        raycasting.addCamera(std::move(cam));
        raycasting.addObject(std::move(board));
        raycasting.addLight(std::move(exampleLight));

        std::vector<Eigen::MatrixXd> images{};
        raycasting.raytraceScene(images);

        if (images.size() != 1) throw std::runtime_error("Expected one rendered image");
        current.m_simulation.validateImage(images.front());

        protocoll.m_simulation.m_simulated_images.push_back(std::move(images.front()));
    };





#endif 
