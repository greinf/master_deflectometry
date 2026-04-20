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
#include "CameraCalibration.hpp"

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
    //Supress open CV Information -only warnings are logged. 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    std::string path{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-16_ReprojektionSimulated/QuantDispCamGamma2.2Lum_ActiveModelBiasNoAper" };

    std::string path_gray_calibration{ "C:/Users/grein/Desktop/Master/Project/deflectometrie/data/2026-04-16_Grauwertkalbirierung_Data_Real/2026-04-16_dark_room" };

    Deflectometry meassure{};

    Pattern& pat = meassure.img_generation();
    ImageProcessing& processing = meassure.processing();

    GrayCodeConfig config{};
    config.creation.inverse = false;
    config.creation.pixel_x = 1920;
    config.creation.pixel_y = 1080;
    config.creation.resolution_x = 500;
    config.creation.resolution_y = 500;
    config.creation.starBit = config.msb;

    std::vector<cv::Mat> grayCode1 = pat.generateGrayCodeImg(config);

    std::size_t grayCodesz{ grayCode1.size() };

    std::vector<cv::Mat> graysequence = pat.generateGrayCalibrationSequence(1);

    std::size_t patternSize{ graysequence.size() };

    std::vector<cv::Mat> images_gt{ grayCode1 };
    for (const auto& img : graysequence) {
        images_gt.push_back(img);
    }

    std::size_t n_pics = grayCodesz + patternSize;

    int n_pics_per_val = 3;

    std::vector<cv::Mat> images = meassure.acquire_img(images_gt, FrameRole::Debug, n_pics_per_val);

    auto it = images.begin();

    std::vector<cv::Mat> meanimg;

    for (std::size_t i = 0; i < n_pics; ++i) {
        auto it_end = std::next(it, n_pics_per_val);
        meanimg.push_back(processing.mean(std::vector<cv::Mat>(it, it_end)));
        it = it_end;
    }

    images = meanimg;

    auto& eval = config.getEvalParameter();
    eval.start = images.begin();
    eval.end = std::next(images.begin(), grayCodesz);

    GrayCodeDecoder dec(processing);

    dec.decoding(config);

    cv::Mat homography =
        processing.createHomographyFromGrayCode(config, cv::Size(1920, 1080));

    std::vector<cv::Mat> maps = processing.createMappingfromHomography(homography, cv::Size(1920, 1080));

    std::vector<cv::Mat> grayValues(std::next(images.begin(), grayCodesz), images.end());

    std::vector<cv::Mat> grayValues_mapped = processing.remapCameraToScreen(grayValues, maps);

    // Active Calibration
    cv::Mat mask_active = cv::Mat::ones(grayValues_mapped[0].size(), CV_8U);

    int border = 50;

    // oben
    mask_active(cv::Range(0, border), cv::Range::all()) = 0;

    // unten
    mask_active(cv::Range(mask_active.rows - border, mask_active.rows), cv::Range::all()) = 0;

    // links
    mask_active(cv::Range::all(), cv::Range(0, border)) = 0;

    // rechts

    cv::Mat mask_passive = config.results.mask;

    mask_active(cv::Range::all(), cv::Range(mask_active.cols - border, mask_active.cols)) = 0;

    // Just make the mask a bit smaller 
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);

    cv::erode(mask_passive, mask_passive, kernel, { -1,-1 }, 50);

    cv::Mat mask_Lut;

    cv::erode(mask_passive, mask_Lut, kernel, { -1,-1 }, 50);

    meassure.do_grayvalue_calibration(grayValues_mapped, mask_active, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveModel);

    //meassure.do_grayvalue_calibration(grayValues_mapped, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveLut);

    meassure.do_grayvalue_calibration(grayValues, mask_passive, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveModel);

    meassure.do_grayvalue_calibration(grayValues, mask_Lut, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveLut);

    meassure.do_grayvalue_calibration(grayValues_mapped, mask_active, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::ActiveModel_Bias);

    meassure.do_grayvalue_calibration(grayValues, mask_passive, 1, 1, true, path_gray_calibration, _defl_::GrayCal::Method::PassiveModel_Bias);


}

