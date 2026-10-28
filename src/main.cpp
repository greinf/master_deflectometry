#include "RayTracer/SyntheticCalibration.hpp"
#include "RayTracer/CalibrationCallAble.hpp"
#include "RayTracer/ProtocollBatchIO.hpp"

#include <iostream>

//int main() {
//    try {
//        const double magnification =
//            (2464.0 * 0.00345 / 1.5) / (30.0 * 12.0);
//
//        SyntheticCalibration generator;
//
//        generator.build(
//            magnification,
//            { &Protocoll::CameraData::C5028_M },
//            {},                    // Nur Nullverzeichnung
//            5,                    // Anzahl der Ansichten
//            SamplingSetting{ 1, 1 },
//            CV_PI / 4.0,
//            { 12, 9 },               // Quadrate des Schachbretts
//            30.0,                  // Quadratseite in mm
//            16.0,
//            { 2464, 2056 },
//            0.00345,
//            60.0,
//            { 0.0, 0.0 },
//            { 11, 15 },
//            42ULL
//        );
//
//        auto protocols = generator.calibrations();
//
//        Protocoll::BatchIO::renderAll(protocols, Tracing_CallAble);
//
//        protocols.front().m_simulation.showAll();
//
//        return 0;
//    }
//    catch (const std::exception& e) {
//        std::cerr << e.what() << '\n';
//        return 1;
//    }
//}
//
// Replace the current diagnostic main.cpp with this file for this run.
// Requires the corrected RayTracer headers AND corrected CameraCalibration.cpp.
#include "RayTracer/SyntheticCalibration.hpp"
#include "RayTracer/CalibrationCallAble.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

static std::string number(size_t i) {
    std::ostringstream s; s << std::setw(3) << std::setfill('0') << i;
    return s.str();
}

static bool report(const fs::path& path, const MonoCalibrationResult& r,
    const cv::Mat& kGT, const cv::Mat& dGT,
    const std::vector<int>& accepted, const CalibrationConfig& c) {
    cv::FileStorage out(path.string(), cv::FileStorage::WRITE);
    if (!out.isOpened()) throw std::runtime_error("Cannot write " + path.string());
    out << "success" << int(r.success) << "error" << r.errorMessage;
    out << "K_GT" << kGT << "D_GT" << dGT << "K_estimated" << r.cameraMatrix;
    out << "D_estimated" << r.distCoeffs << "RMS" << r.avgReprojectionError;
    out << "optimizer_RMS" << r.optimizerRms << "flags" << r.calibrationFlags;
    out << "accepted_indices" << accepted << "SB" << int(c.useSBDetector);
    out << "kernel_bias" << c.preprocessingKernelBias;
    out << "max_iterations" << c.maxIterations << "epsilon" << c.epsilon;
    std::cout << "\n" << path.filename().string() << ": " << accepted.size()
        << " accepted views; RMS=" << r.avgReprojectionError << '\n';
    if (r.success) {
        cv::Mat delta = r.cameraMatrix - kGT;
        out << "delta_K" << delta;
        std::cout << "K_GT:\n" << kGT << "\nK:\n" << r.cameraMatrix
            << "\ndelta_K:\n" << delta << '\n';
    }
    else std::cerr << r.errorMessage << '\n';
    return r.success;
}

int main() {

    std::filesystem::path path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-10-28_DistortionCalibration/Test/test.csv" };

    CameraCalibration::_TestDistortionCalibrationConfig config{};
    config.add_noise = false;
    config.stddev_noise = 0.4;
    config.p1 = 0.1;
    config.p2 = 0.1;

    CameraCalibration::_TestDistortionCalibration(config, {}, path);


    //try {
    //    // EDIT THIS ONE PATH. Each run creates a NEW subdirectory.
    //    const fs::path root = R"(C:\Users\grein\Desktop\SyntheticCalibration_50mm)";
    //    constexpr size_t views = 10;
    //    const cv::Size imageSize{ 2464, 2056 };
    //    const SamplingSetting samples{ 1, 1 };
    //    const double magnification = (2464.0 * 0.00345 / 1.75) / (30.0 * 12.0);

    //    SyntheticCalibration generator;
    //    // Only 50 mm, zero distortion: no matching to another objective.
    //    generator.build(magnification, { &Protocoll::CameraData::C5028_M }, {},
    //        views, samples, CV_PI / 4.0, { 12, 9 }, 30.0, 16.0,
    //        { imageSize.width, imageSize.height }, 0.00345, 60.0,
    //        { 0.0, 0.0 }, { 11, 15 }, 42ULL);
    //    auto protocol = generator.calibrations().front();
    //    fs::create_directories(root);
    //    fs::path directory;
    //    for (size_t i = 0;; ++i) {
    //        directory = root / ("C5028_M_4x4_" + number(i));
    //        if (fs::create_directory(directory)) break;
    //    }
    //    std::cout << "Output: " << directory << '\n';
    //    // Saves XML and validates V2 metadata/platform before streaming images.
    //    protocol.saveProtocol(directory);
    //    fs::create_directory(directory / "previews");

    //    cv::Mat kGT, dGT;
    //    cv::eigen2cv(protocol.m_camera.intrinsicMatrix, kGT);
    //    cv::eigen2cv(protocol.m_camera.distortionCoefficients, dGT);
    //    CalibrationConfig c;
    //    c.boardSize = { 11, 8 }; c.squareSize = 30.0f;
    //    c.zeroTangentDist = true; c.fixK1 = c.fixK2 = c.fixK3 = true;
    //    c.fixAspectRatio = false; c.useIntrinsicGuess = true;
    //    c.preprocessingKernelBias = 2; // s=4 -> odd Gaussian kernel 7, then AREA.
    //    auto exactPoints = SyntheticCalibration::projectGroundTruthCorners(protocol);
    //    std::vector<int> all;
    //    for (size_t i = 0; i < views; ++i) all.push_back(static_cast<int>(i));
    //    auto exact = CameraCalibration::calibrateMonoSyn(exactPoints, c, kGT, dGT, imageSize);
    //    if (!report(directory / "exact.xml", exact, kGT, dGT, all, c))
    //        throw std::runtime_error("Exact calibration failed; rendering cancelled");

    //    // V2 streaming uses the same layout as corrected Simulation::save().
    //    // Keep one raw image in RAM; do not retain 20 raw 4x4 images.
    //    const fs::path partial = directory / "Binary.bin.partial";
    //    std::ofstream binary(partial, std::ios::binary);
    //    if (!binary) throw std::runtime_error("Cannot open raw image output");
    //    binary << "RayTracerSimulationV2\n" << samples.samples_x << ' '
    //        << samples.samples_y << ' ' << imageSize.width << ' '
    //        << imageSize.height << " 1 " << views
    //        << "\nIEEE754_LE_COLUMN_MAJOR_F64\n";
    //    std::vector<std::vector<cv::Point2f>> points[2];
    //    std::vector<int> indices[2];
    //    for (size_t i = 0; i < views; ++i) {
    //        std::cout << "\nRendering " << i + 1 << '/' << views << "\n" << std::flush;
    //        auto& rawImages = protocol.m_simulation.m_simulated_images;
    //        rawImages.clear();
    //        Tracing_CallAble(protocol, 0, i);
    //        if (rawImages.size() != 1) throw std::runtime_error("Expected one image");
    //        const auto& raw = rawImages.front();
    //        protocol.m_simulation.validateImage(raw);
    //        binary << "BinaryImage\n" << raw.rows() << ' ' << raw.cols() << " double\n";
    //        binary.write(reinterpret_cast<const char*>(raw.data()),
    //            static_cast<std::streamsize>(raw.size()) * sizeof(double));
    //        binary << "\n\n";
    //        binary.flush();
    //        if (!binary) throw std::runtime_error("Raw image write failed (disk full?)");
    //        cv::Mat image;
    //        cv::eigen2cv(protocol.m_simulation.prepareImageforProcessing(raw,
    //            c.preprocessingKernelBias), image);
    //        image = CameraCalibration::toGray8(image);
    //        const auto preview = directory / "previews" / ("image_" + number(i) + ".png");
    //        if (!cv::imwrite(preview.string(), image)) throw std::runtime_error("PNG write failed");
    //        for (int detector = 0; detector < 2; ++detector) {
    //            std::vector<cv::Point2f> corners;
    //            const bool found = CameraCalibration::detectChessboardCorners(
    //                image, c.boardSize, corners, c.subPixWindow, detector == 1);
    //            const std::string name = detector ? "SB" : "classic";
    //            std::cout << name << ": " << (found ? "OK" : "FAILED") << '\n';
    //            if (found) { indices[detector].push_back(static_cast<int>(i)); points[detector].push_back(corners); }
    //            cv::Mat overlay; cv::cvtColor(image, overlay, cv::COLOR_GRAY2BGR);
    //            cv::drawChessboardCorners(overlay, c.boardSize, corners, found);
    //            if (!cv::imwrite((directory / "previews" /
    //                (name + "_" + number(i) + ".png")).string(), overlay))
    //                throw std::runtime_error("Overlay write failed");
    //        }
    //        rawImages.clear();
    //    }
    //    binary.close();
    //    if (!binary) throw std::runtime_error("Raw image close failed");
    //    fs::remove(directory / "Binary.bin"); // replace only our own empty placeholder
    //    fs::rename(partial, directory / "Binary.bin");
    //    bool anySuccess = false;
    //    for (int detector = 0; detector < 2; ++detector) {
    //        c.useSBDetector = detector == 1;
    //        auto result = CameraCalibration::calibrateMonoSyn(points[detector], c, kGT, dGT, imageSize);
    //        anySuccess |= report(directory / (detector ? "images_SB.xml" : "images_classic.xml"),
    //            result, kGT, dGT, indices[detector], c);
    //    }
    //    std::cout << "\nSaved V2 dataset and previews: " << directory << '\n';
    //    return anySuccess ? 0 : 2;
    //}
    //catch (const std::exception& e) {
    //    std::cerr << "ERROR: " << e.what() << '\n';
    //    return 1;
    //}
}


