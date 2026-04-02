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


struct GrayCodeSet {
    // index 0 corresponds to 1 (Scr01) if your files start at 01
    std::vector<cv::Mat> x;
    std::vector<cv::Mat> y;

    // Keep original names so we can save with identical filenames
    std::vector<std::string> xNames;
    std::vector<std::string> yNames;
};

static bool ensureDirExists(const std::filesystem::path& p) {
    std::error_code ec;
    if (std::filesystem::exists(p, ec)) return std::filesystem::is_directory(p, ec);
    return std::filesystem::create_directories(p, ec);
}

GrayCodeSet loadGrayCodeFromFolder(const std::filesystem::path& folder, bool asGray = true) {
    if (!std::filesystem::exists(folder) || !std::filesystem::is_directory(folder))
        throw std::runtime_error("Folder does not exist or is not a directory: " + folder.string());

    // Matches:
    // GrayCodeXScr01.bmp
    // GrayCodeYScr11.bmp
    // also allows .png/.jpg if you want to extend (currently bmp only, change if needed)
    const std::regex re(R"(GrayCode([XY])Scr(\d+)\.bmp$)", std::regex::icase);

    struct Entry { char axis; int idx; std::filesystem::path path; std::string name; };
    std::vector<Entry> entries;

    for (const auto& de : std::filesystem::directory_iterator(folder)) {
        if (!de.is_regular_file()) continue;

        const std::string fname = de.path().filename().string();
        std::smatch m;
        if (std::regex_match(fname, m, re)) {
            char axis = static_cast<char>(std::toupper(m[1].str()[0])); // 'X' or 'Y'
            int idx = std::stoi(m[2].str());                           // e.g. 1..11
            entries.push_back({ axis, idx, de.path(), fname });
        }
    }

    if (entries.empty())
        throw std::runtime_error("No GrayCode[X|Y]ScrXX.bmp files found in: " + folder.string());

    // Sort by axis, then idx
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.axis != b.axis) return a.axis < b.axis;
        return a.idx < b.idx;
        });

    // Determine max index for X and Y (so we can size vectors)
    int maxX = 0, maxY = 0;
    for (const auto& e : entries) {
        if (e.axis == 'X') maxX = std::max(maxX, e.idx);
        else               maxY = std::max(maxY, e.idx);
    }

    GrayCodeSet set;
    set.x.resize(maxX);
    set.y.resize(maxY);
    set.xNames.resize(maxX);
    set.yNames.resize(maxY);

    const int imreadFlag = asGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED;

    for (const auto& e : entries) {
        cv::Mat img = cv::imread(e.path.string(), imreadFlag);
        if (img.empty())
            throw std::runtime_error("Failed to read image: " + e.path.string());

        // idx is 1-based in filename -> store at idx-1
        const int pos = e.idx - 1;
        if (pos < 0) throw std::runtime_error("Invalid index in filename: " + e.name);

        if (e.axis == 'X') {
            if (pos >= static_cast<int>(set.x.size()))
                throw std::runtime_error("Index out of range for X: " + e.name);
            set.x[pos] = img;
            set.xNames[pos] = e.name;
        }
        else {
            if (pos >= static_cast<int>(set.y.size()))
                throw std::runtime_error("Index out of range for Y: " + e.name);
            set.y[pos] = img;
            set.yNames[pos] = e.name;
        }
    }

    
    return set;
}

static std::string makeName(char axis, size_t i) {
    std::ostringstream oss;
    oss << "GrayCode" << axis << "Scr"
        << std::setw(2) << std::setfill('0') << (i + 1)
        << ".bmp";
    return oss.str();
}

void saveGrayCodeToFolder(const GrayCodeSet& set, const std::filesystem::path& outFolder) {
    if (!ensureDirExists(outFolder))
        throw std::runtime_error("Could not create output directory: " + outFolder.string());

    // Save X
    for (size_t i = 0; i < set.x.size(); ++i) {
        if (set.x[i].empty()) continue;

        const std::filesystem::path outPath = outFolder /
            (set.xNames[i].empty() ? makeName('X', i) : set.xNames[i]);

        if (!cv::imwrite(outPath.string(), set.x[i]))
            throw std::runtime_error("Failed to write: " + outPath.string());
    }

    // Save Y
    for (size_t i = 0; i < set.y.size(); ++i) {
        if (set.y[i].empty()) continue;

        const std::filesystem::path outPath = outFolder /
            (set.yNames[i].empty() ? makeName('Y', i) : set.yNames[i]);

        if (!cv::imwrite(outPath.string(), set.y[i]))
            throw std::runtime_error("Failed to write: " + outPath.string());
    }
}



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


int main()
{

    std::filesystem::path inFolder = R"(C:\Users\grein\Desktop\Data)";
    std::filesystem::path outFolder = R"(C:\Users\grein\Desktop\Data_outBigDistance)";
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/1Wave" };

    std::string path_gray_sections{ "C:/Users/grein/Desktop/2026-02-15_GrayCalibSections2.csv" };

    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-15_GrayCalibSections" };
    //Path to IDS calibration: Blende geschlossen. 
    std::string camMatrix_path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-09_StereoMAKO.xml" };

    std::string camMatrix_path1{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-09_StereoMAKO" };


    std::string calibPath{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2026-02-09StereoCalibration" };

    std::string calibrationDisplayCampath{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-02-17_Display_Cam" };

    Deflectometry meassure{};
    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    GrayCodeConfig config{};
    config.creation.inverse = false;
    config.creation.pixel_x = 500;
    config.creation.pixel_y = 500;
    config.creation.resolution_x = 500;
    config.creation.resolution_y = 500;
    config.creation.starBit = config.msb;
    
    std::vector<cv::Mat> grayCode1 = pat.generateGrayCodeImg(config);

    GrayCodeDecoder dec(processing);

    std::vector<cv::Mat> decoding = dec.decoding(config);
    
    cv::Mat homography = 
        processing.createHomographyFromGrayCode(config, cv::Size(1920, 1080));

    std::vector<cv::Mat> maps = processing.createMappingfromHomography(homography, { 500, 500 });

    std::vector<cv::Mat> grayCodemapped = processing.remapCameraToScreen(grayCode1, maps);


    
    meassure.GrayCalibrationClassTest(_defl_::GrayCal::Method::ActiveLut);

    // AbstandsMessung 
    meassure.load(FrameRole::CalibrationMatrix, camMatrix_path);
    std::vector<cv::Mat> camMatrix = meassure.get(FrameRole::CalibrationMatrix);

    meassure.load(FrameRole::CalibCamToCam, camMatrix_path);
    std::vector<cv::Mat> camToCam = meassure.get(FrameRole::CalibCamToCam);

    std::vector<cv::Mat> dist_Coeffs = meassure.get(FrameRole::DistortionCoeff);

    std::optional<std::vector<cv::Mat>> cameramatrix_distCoeff;
    cameramatrix_distCoeff.emplace({ camMatrix[0], dist_Coeffs[0] });

    std::vector<cv::Mat> real_pattern =
        meassure.createSyntheticalImages(
            Warping::raycasting,                 // Warping: homography-based warping

            cameramatrix_distCoeff,              // camera_matrix (std::optional<std::vector<cv::Mat>>)

            2.2,                                 // gamma: display / camera gamma

            true,                                // display_quantize: quantize display output

            true,                                // camera_quantization: simulate camera ADC quantization

            true,                                // luminance: apply luminance weighting

            true,                                // smoothing: enable optical smoothing

            true,                                // warp: apply geometric warping

            200.0,                               // image_height: mirror circumference [mm]

            2464,                                // dest_width: Mako G-507-B sensor width (px)

            2056,                                // dest_height: Mako G-507-B sensor height (px)

            0.2745,                              // display_pixel_pitch: display pixel pitch [mm]

            0.0,                                 // display_shift_x: lateral display shift x [mm]

            0.0,                                 // display_shift_y: lateral display shift y [mm]

            0.0,                                 // display_tilt_x: display tilt around x-axis [rad]

            0.0,                                 // display_tilt_y: display tilt around y-axis [rad]

            Shift_mode::four_phase_shift,        // mode: phase-shift pattern mode

            _defl_::GrayCal::Method::None,       // method: no gray-value calibration

            1920,                                // pattern_width: display width (px)

            1080,                                // pattern_height: display height (px)

            10,                                  // n_periods_in_y: number of sinusoidal periods in y

            2.4,                                 // aperture_number: f-number (N)

            2000,                                // distance: camera–mirror distance [mm] (2*f, f=1600)

            6.6,                                 // object_height: sensor height [mm] (2/3" → 6.6 mm)
            
            RoiBorders<double>{                  // destination ROI in destination image
                {200.0, 200.0},                  // left-up corner 
                { 1800.0, 180.0 },                 // left-down corner
                { 210.0, 2200.0 },                 // right-up corner
                { 1900.0, 1900.0 }                 // right-down corner
            },
            
            calibPath                            // path to the stored GrayCalibration values
            
        );

    meassure.testReprojection(
        camMatrix_path,
        path_gray_calibration,
        path,
        Shift_mode::four_phase_shift,
        _defl_::GrayCal::Method::ActiveLut,
        33.75,
        false,
        UnwrapMode::opencv,
        true);

    
    //// ************ Triangulation *******************

    //auto result = meassure.computeLaserDistanceFrom4Images(
    //    LaserFr[0], LaserFr[1], LaserFr[2], LaserFr[3],
    //    camMatrix[0], dist_Coeffs[0], camMatrix[1], dist_Coeffs[1], camToCam[0], camToCam[1]
    //);

    //std::cout << "p1 (px): " << result.p1_px << "\n";
    //std::cout << "p2 (px): " << result.p2_px << "\n";
    //std::cout << "3D (cam1): [" << result.X_cam1.x << ", " << result.X_cam1.y << ", " << result.X_cam1.z << "]\n";
    //std::cout << "distance to cam1: " << result.distance_cam1 << " (same units as T)\n";

    GrayCodeSet gc = loadGrayCodeFromFolder(inFolder, true);

    std::vector<cv::Mat> grayCodeXY;

    for (const auto& img : gc.x) {
        grayCodeXY.push_back(img);
    }

    for (const auto& img : gc.y) {
        grayCodeXY.push_back(img);
    }

    std::vector<cv::Mat> grayCode = meassure.acquire_img(grayCodeXY, FrameRole::Debug, 1);
    
    gc.x = std::vector<cv::Mat>(grayCode.begin(), std::next(grayCode.begin(), 11));
    gc.y = std::vector<cv::Mat>(std::next(grayCode.begin(),11), grayCode.end());



    double pixelpitch = 0.2745;

    std::vector<cv::Mat> pattern10 =
        meassure.generatePattern(Shift_mode::four_phase_shift, _defl_::GrayCal::Method::None, path_gray_calibration, FrameRole::Debug, false, " ", 10);

    
    std::vector<cv::Mat> wrappedPhase =
        meassure.do_wrapped_phase(pattern10, 1, 4, true, path);

    //meassure.load(FrameRole::Contrast, path);
    std::vector<cv::Mat> contrastPhase = meassure.get(FrameRole::Contrast);

    cv::Mat mask = meassure.getMask(contrastPhase, 0.0, false);

    std::vector<cv::Mat> unwrappedPhase =
        meassure.do_unwrapped_phase(wrappedPhase, mask, UnwrapMode::manually, true, path, 108);

    cv::Mat reference_img =
        meassure.generate_reference_Pattern(ReferenceMode::checkerboard, (int)4, true, path);

   /* std::vector<cv::Mat> reference =
        meassure.getFrames(FrameRole::Debug, 1, reference_img);*/

    mask = meassure.getMask(contrastPhase, 0.0, true);

    std::vector<cv::Vec2d> ref_point =
        meassure.getReferencePoint(
            std::vector<cv::Mat>{reference_img},
            ReferenceMode::checkerboard, 
            mask);

    std::vector<cv::Mat> pattern_distorted;
    for (auto& img : unwrappedPhase) {
        pattern_distorted.emplace_back(meassure.distortImage_manual(img, camMatrix[0], dist_Coeffs[0]));
    }


    std::vector<cv::Mat> reprojection = meassure.do_reprojection(
        pattern_distorted,
        mask,
        ref_point[0],
        camMatrix[0],
        dist_Coeffs[0],
        108.0,
        unwrappedPhase[0].cols,
        unwrappedPhase[0].rows,
        0.2745, //PixelPitch  FH 0.277    BMZ: 
        true,
        path,
        532, //Dispaly Width  FH    BMZ: 527.04
        299.2 //Dispaly Height FH:      BMZ: 296.46
    );


    mask = meassure.getMask(contrastPhase, 0.3, true);

    std::vector<cv::Mat> reprojection1 = meassure.do_reprojection(
        unwrappedPhase,
        mask,
        ref_point[0],
        camMatrix[0],
        dist_Coeffs[0],
        108.0,
        unwrappedPhase[0].cols,
        unwrappedPhase[0].rows,
        0.2745, //PixelPitch  FH 0.277    BMZ: 
        true,
        path,
        532, //Dispaly Width  FH    BMZ: 527.04
        299.2 //Dispaly Height FH:      BMZ: 296.46
    );

    /*cv::Mat distortionError =
        meassure.calcDistortionError(distorted);

    cv::Mat undistortionError =
        meassure.calcDistortionError(undistorted);

    meassure.saveSingleImage(FrameRole::DistortErrX, distortionError, true, path);

    meassure.saveSingleImage(FrameRole::DistortErrX, undistortionError, true, path);*/


    //meassure.loadPhaseConfig("C:/Users/grein/Desktop/Master/Project/deflectometrie/out/2025-11-23/0.xml");

}
