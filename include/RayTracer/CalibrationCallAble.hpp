#ifndef CALLIBRATIONCALLABLE_HPP	
#define CALLIBRATIONCALLABLE_HPP

#include "Scene.hpp"
#include "Utils.hpp"
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include <iostream>

auto Tracing_CallAble = [](
    Protocoll::SimulationProtocoll& protocoll,
    const std::size_t setupIndex,
    const std::size_t poseIndex) -> void
    {
        // Create Objects
        const auto& current = protocoll;

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

        protocoll.m_simulation.m_simulated_images.push_back(images.front());
    };





#endif 