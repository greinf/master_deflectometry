#include "imgProcessing.hpp"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <math.h>
#include <imageHandler.hpp>
#include <opencv2/phase_unwrapping/histogramphaseunwrapping.hpp>
#include "GoldsteinWrapper.hpp"
#include <filesystem>


ImageProcessing::ImageProcessing() {
    ++instance_counter;
    if (instance_counter > 1) throw std::runtime_error("Only one instance of ImageProcessing allowed");
    
}

// create() initializes the arrays for the processing. This methods needs to be called one time before further processsing. 
void ImageProcessing::create(std::vector<cv::Mat>& vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            std::cerr << "All pictures must be same kind and type. \n";
            throw std::runtime_error("All pictures must be same kind and type. \n");
        }
    }
    for (auto& m : m_s1) m = cv::Mat(vec.at(0).size(), CV_32F, cv::Scalar(0.0));
    for (auto& m : m_s2) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_s3) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_baseIntensity) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_contrast) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_wrapped_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
    for (auto& m : m_unwrapped_phase) m = cv::Mat::zeros(runtime_flags.pixel_y, runtime_flags.pixel_x, CV_32F);
}

void ImageProcessing::bayerToGray()
{
    for (const auto& frame : m_phase) {
		cv::cvtColor(frame, frame, cv::COLOR_BayerRG2GRAY);
    }
}

//Debugging function to calculate gradient strength
auto grad_strength = [](const cv::Mat& f) -> std::pair<double, double> {
    cv::Mat fx, fy;
    cv::Sobel(f, fx, CV_32F, 1, 0, 3); // d/dx
    cv::Sobel(f, fy, CV_32F, 0, 1, 3); // d/dy
	cv::imshow("Sobel X", fx);
	cv::imshow("Sobel Y", fy);
    cv::waitKey();
    return std::pair<double, double>(
        cv::mean(cv::abs(fx))[0],
        cv::mean(cv::abs(fy))[0]
    );
    };

void ImageProcessing::saveImages(std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    
    if (fs.isOpened()) {
        fs << "wrappedPhasehorizontal" << m_wrapped_phase.at(0);
        fs << "wrappedPhasevertical" << m_wrapped_phase.at(1);
        fs << "contrasthorizontal" << m_contrast.at(0);
        fs << "contrastvertical" << m_contrast.at(1);
        fs << "baseIntensityhorizontal" << m_baseIntensity.at(0);
        fs << "baseIntensityvertical" << m_baseIntensity.at(1);
        fs << "unwrappedPhasehorizontal" << m_unwrapped_phase.at(0);
        fs << "unwrappedPhasevertical" << m_unwrapped_phase.at(1);
    }
}

ImageProcessing::~ImageProcessing() {
    --instance_counter;
}
void ImageProcessing::wrapped_phase() {
    
    create(m_frames);
	std::cout << "Datatype " << m_frames[0].type() << '\n';
    
    if (runtime_flags.get_number_of_pictures_per_pattern() <= 0) {
        std::cerr << "Flag how many Picutres per Pattern are created must be set to specific value != 0 \n";
        return;
    }

    if (runtime_flags.get_number_of_pictures_per_pattern() > 1) {
        int n_expected_frames = runtime_flags.get_number_of_pictures_per_pattern() *
            runtime_flags.get_number_of_shifts() * 2;
        std::cout << "Expected " << n_expected_frames << '\n' <<
            "std::vector size " << m_frames.size() << '\n';

        assert(n_expected_frames == m_frames.size() && "For valid Processing, number of expected Frames (calculated from flagHandler.hpp\
			 flags) must match vector size \n");
        std::vector<cv::Mat>::iterator begin = m_frames.begin();
        
        for (int i = 0; i < (runtime_flags.get_number_of_shifts()*2); ++i) {
            std::vector<cv::Mat>::iterator end = begin + runtime_flags.get_number_of_pictures_per_pattern();
            m_raw_phase.push_back(mean(std::vector<cv::Mat>(begin, end)));
            begin = end;
        }
    }

    if (runtime_flags.get_number_of_pictures_per_pattern() == 1) {
        m_raw_phase = m_frames;
    }
    
    for (auto& frame : m_raw_phase) {
        if (frame.type() != CV_32F)
            frame.convertTo(frame, CV_32F, 1.0f / 255.0f);
    }

    for (int i = 0; i < runtime_flags.get_number_of_shifts(); ++i) {    
        float phase = static_cast<float>((CV_2PI * i) / runtime_flags.get_number_of_shifts());
        //m_s1.at(0) += m_raw_phase.at(i).forEach<float>([&](float& a, const int* position) -> void {
        //    a = a * std::sin(phase); });
        m_s1.at(0) += m_raw_phase.at(i) * std::sinf(phase);
        m_s1.at(1) += m_raw_phase.at(i + runtime_flags.get_number_of_shifts()) * std::sinf(phase);
        m_s2.at(0) += m_raw_phase.at(i) * std::cosf(phase);
        m_s2.at(1) += m_raw_phase.at(i + runtime_flags.get_number_of_shifts()) * std::cosf(phase);
        m_s3.at(0) += m_raw_phase.at(i);
        m_s3.at(1) += m_raw_phase.at(i + runtime_flags.get_number_of_shifts());
    }

    for (int i = 0; i < m_baseIntensity.size(); ++i) {
        m_baseIntensity.at(i) = m_s3.at(i) / runtime_flags.get_number_of_shifts();
        //Initialize already a array with the right datatype and size()
    }

    for (int i = 0; i < m_s1.at(0).rows; ++i) {
        for (int j = 0; j < m_s1.at(0).cols; ++j) {
            m_wrapped_phase.at(0).at<float>(i, j) = std::atan2f(m_s1.at(0).at<float>(i, j), m_s2.at(0).at<float>(i, j));
            m_wrapped_phase.at(1).at<float>(i, j) = std::atan2f(m_s1.at(1).at<float>(i, j), m_s2.at(1).at<float>(i, j));
            m_contrast.at(0).at<float>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(0).at<float>(i, j), 2) + std::pow(m_s2.at(0).at<float>(i, j), 2))) / m_s3.at(0).at<float>(i, j);
            m_contrast.at(1).at<float>(i, j) = (2 * std::sqrt(std::pow(m_s1.at(1).at<float>(i, j), 2) + std::pow(m_s2.at(1).at<float>(i, j), 2))) / m_s3.at(1).at<float>(i, j);
        }
    }
    
    double phaseminVal, phasemaxVal, baseminVal, basemaxval, conminval, conmaxval;
    cv::Point phaseminloc, phasemaxloc, baseminloc, basemaxloc, conminloc, conmaxloc;
    
    /*Debugging*/
    
    cv::minMaxLoc(m_wrapped_phase[1],&phaseminVal, &phasemaxVal);
    cv::minMaxLoc(m_baseIntensity[1], &baseminVal, &basemaxval);
    cv::minMaxLoc(m_contrast[1], &conminval, &conmaxval);

    std::cout << "Phase: " << phaseminVal << " … " << phasemaxVal << '\n'
        << "Base intensity: " << baseminVal << " … " << basemaxval << '\n'
        << "Contrast: " << conminval << " … " << conmaxval << '\n';

    cv::Mat wrapped8cu1, wrapped8cu2, constrast8cu1, contrast8cu2;
    cv::normalize(m_wrapped_phase[0], wrapped8cu1, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_wrapped_phase[1], wrapped8cu2, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_contrast[0], constrast8cu1, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_contrast[1], contrast8cu2, 0, 255, cv::NORM_MINMAX, CV_8U);
    std::filesystem::path outDir("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out");
    if(std::filesystem::exists(outDir) == false) {
        std::filesystem::create_directory(outDir);
	}
	std::filesystem::path path1 = outDir / "wrapped_phase1.png";
	std::filesystem::path path2 = outDir / "wrapped_phase2.png";
	std::filesystem::path path3 = outDir / "base_intensity1.png";
	std::filesystem::path path4 = outDir / "contrast1.png";
	std::filesystem::path path5 = outDir / "base_intensity2.png";
	std::filesystem::path path6 = outDir / "contrast2.png";

    try {
        cv::imwrite(path1.string(), wrapped8cu1);
        cv::imwrite(path2.string(), wrapped8cu2);
		cv::imwrite(path4.string(), constrast8cu1);
        cv::imwrite(path6.string(), contrast8cu2);

    }
    catch (std::exception& e) { std::cout << "Excpetion " << e.what() << std::endl; }
    
    //Debugging
    /*
    auto [Hx, Hy] = grad_strength(m_wrapped_phase[0]); // horizontal set
    auto [Vx, Vy] = grad_strength(m_wrapped_phase[1]); // vertical set
    std::cout << "Wrapped H: |dx|=" << Hx << " |dy|=" << Hy << "\n";
    std::cout << "Wrapped V: |dx|=" << Vx << " |dy|=" << Vy << "\n";
    */

    /*
    cv::imshow("Contrast", m_contrast.at(0));
    cv::imshow("Base Intensity", m_baseIntensity.at(0));
    cv::imshow("Phase", m_phase.at(0));

    cv::Mat phaseVis, contrastVis;
    cv::normalize(m_phase.at(0), phaseVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_contrast.at(0), contrastVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::applyColorMap(phaseVis, phaseVis, cv::COLORMAP_JET);
    cv::applyColorMap(contrastVis, contrastVis, cv::COLORMAP_JET);

    cv::imshow("Phase (color)", phaseVis);
    cv::imshow("Contrast (color)", contrastVis);
    cv::waitKey(0);
    */
    /*Just to see results*/
    /*
    std::thread img(&ImageHandler::run, imgHandler, 3);
    imgHandler.imshow_Camera(m_baseIntensity.at(0));
    imgHandler.imshow_Pattern(m_phase.at(0));
    imgHandler.imshow_Processed(m_contrast.at(0));
    std::this_thread::sleep_for(std::chrono::seconds(10));
    imgHandler.stop();
    if (img.joinable()) img.join();
    */
}

void ImageProcessing::goldsteinUnwrap() {
    cv::Mat unwrapped1;
    cv::Mat unwrapped2;
    for (auto& m : m_unwrapped_phase) {
		cv::normalize(m, m, 0, 1, cv::NORM_MINMAX, CV_32F); 
    }
    cv::Mat mask1 = (m_contrast[0] > 0.2f); // for example: keep only valid contrast regions
    mask1.convertTo(mask1, CV_8U);           // ensure binary mask
	cv::Mat mask2 = (m_contrast[1] > 0.2f);
	mask2.convertTo(mask2, CV_8U);
    goldsteinUnwrapCV(m_wrapped_phase[0], unwrapped1, mask1);
	goldsteinUnwrapCV(m_wrapped_phase[1], unwrapped2, mask2);

	cv::normalize(unwrapped1, m_unwrapped_phase[0], 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::normalize(unwrapped2, m_unwrapped_phase[1], 0, 255, cv::NORM_MINMAX, CV_8U);    
	cv::imshow("Goldstein Unwrapped Phase 1", m_unwrapped_phase.at(0));
	cv::imshow("Goldstein Unwrapped Phase 2", m_unwrapped_phase.at(1));
    cv::waitKey(0);

    /*
    goldsteinUnwrapCV(m_phase.at(0), m_unwrapped_phase.at(0));
	cv::imshow("Goldstein Unwrapped Phase", m_unwrapped_phase.at(0));
	goldsteinUnwrapCV(m_phase.at(1), m_unwrapped_phase.at(1));
	cv::imshow("Goldstein Unwrapped Phase 2", m_unwrapped_phase.at(1));
	cv::waitKey(0);
    
    double phaseminVal, phasemaxVal, baseminVal, basemaxval, conminval, conmaxval;

    cv::minMaxLoc(m_unwrapped_phase[0], &phaseminVal, &phasemaxVal);
    cv::minMaxLoc(m_unwrapped_phase[1], &baseminVal, &basemaxval);

    std::cout << "Phase1: " << phaseminVal << " … " << phasemaxVal << '\n'
        << "Phase 2: " << baseminVal << " … " << basemaxval << '\n';
        */
}


void ImageProcessing::unwrapped_phase() {
    //std::cout << "Phase unwrapping \n";
    cv::Mat unwrappedPhase1, unwrappedPhase2;
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params params;
    params.height = runtime_flags.pixel_y;
    params.width = runtime_flags.pixel_x;
    params.histThresh = CV_PI / 10; // you can tune this threshold
    params.nbrOfSmallBins = 20;
    params.nbrOfLargeBins = 10;
    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> unwrapping = cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);
    cv::Mat mask1 = (m_contrast.at(0) > 0.1f);  // choose threshold as needed
	cv::Mat mask2 = (m_contrast.at(1) > 0.1f);  // choose threshold as needed
    unwrapping->unwrapPhaseMap(m_wrapped_phase.at(0), m_unwrapped_phase.at(0), mask1);

    // vertical unwrap, rotated to horizontal orientation
    cv::Mat wrappedTransposed, maskTransposed, unwrappedTransposed;
    cv::transpose(m_wrapped_phase.at(1), wrappedTransposed);
    cv::transpose(m_contrast.at(1), maskTransposed);
    maskTransposed = (maskTransposed > 0.3f);

    // create the same params (same pixel_y/pixel_x, do NOT swap)
    params.height = wrappedTransposed.rows;
    params.width = wrappedTransposed.cols;
    params.histThresh = CV_PI / 10;
    params.nbrOfSmallBins = 20;
    params.nbrOfLargeBins = 10;

    auto unwrapping1 = cv::phase_unwrapping::HistogramPhaseUnwrapping::create(params);

    // unwrap the *transposed* image
    unwrapping1->unwrapPhaseMap(wrappedTransposed, unwrappedTransposed, maskTransposed);

    // transpose back
    cv::transpose(unwrappedTransposed, unwrappedPhase2);
    m_unwrapped_phase.at(1) = unwrappedPhase2;
    cv::Mat unwrappedPhase1_8u, unwrappedPhase2_8u;

    cv::normalize(m_unwrapped_phase.at(0), unwrappedPhase1_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_unwrapped_phase.at(1), unwrappedPhase2_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imshow("Unwrapped Phase 1", unwrappedPhase1_8u);
    cv::imshow("Unwrapped Phase 2", unwrappedPhase2_8u);
    cv::waitKey(0);
    /*
    std::filesystem::path outDir("C:\\Users\\grein\\Desktop\\Master\\Project\\deflectometrie\\out");
    if (std::filesystem::exists(outDir) == false) {
        std::filesystem::create_directory(outDir);
    }
    std::filesystem::path path1 = outDir / "unwrapped_phase1.png";
    std::filesystem::path path2 = outDir / "unwrapped_phase2.png";
    cv::imwrite(path1.string(), unwrappedPhase1_8u);
    cv::imwrite(path2.string(), unwrappedPhase2_8u);
    */
}

cv::Mat ImageProcessing::mean(std::vector<cv::Mat> vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            std::cerr << "All pictures must be same kind and type. \n";
            throw std::runtime_error ("All pictures must be same kind and type. \n");
        }
    }
    cv::Mat acc;
    vec[0].convertTo(acc, CV_32FC1);

    for (size_t i = 1; i < vec.size(); ++i) {
        cv::Mat temp;
        vec[i].convertTo(temp, CV_32FC1);
        acc += temp;   // pixelweise Addition
    }

    acc /= static_cast<float>(vec.size());  // pixelweise Division

    // Zurück zu 8 Bit
    cv::Mat average;
    acc.convertTo(average, CV_8UC1);
    //Debugging
	//cv::imshow("Mean Image", average);
    //cv::waitKey();
    return average;
}


cv::Mat ImageProcessing::load_images(std::string path) {
    std::filesystem::path image_location{ path };
    if (std::filesystem::exists(image_location)) {
        return cv::imread(path, cv::ImreadModes::IMREAD_GRAYSCALE);
    }
}

