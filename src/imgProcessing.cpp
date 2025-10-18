#include "imgProcessing.hpp"

#include <opencv2/opencv.hpp>
#include <cassert>

void ImageProcessing::wrapped_phase(std::vector<cv::Mat>& vec) {
    if (runtime_flags.get_number_of_pictures_per_pattern() <= 0) {
        std::cerr << "Flag how many Picutres per Pattern are created must be set to specific value != 0 \n";
        return;
    }

    if (runtime_flags.get_number_of_pictures_per_pattern() > 1) {
        int n_expected_frames = runtime_flags.get_number_of_pictures_per_pattern() *
            runtime_flags.get_number_of_shifts();
        assert(n_expected_frames == vec.size() && "For valid Processing, number of expected Frames (calculated from flagHandler.hpp\
			 flags) must match vector size \n");
        std::vector<cv::Mat>::iterator begin = vec.begin();
        std::vector<cv::Mat>::iterator end = begin + 5;
        for (int i = 0; i < runtime_flags.get_number_of_shifts() * 2; ++i) {
            m_raw_phase.push_back(mean(std::vector<cv::Mat>(begin, end)));
            std::advance(begin, 5);
            std::advance(end, 5);
        }
    }

    if (runtime_flags.get_number_of_pictures_per_pattern() == 1) {
        m_raw_phase = vec;
    }
    for (auto& m : std::initializer_list <const char*> { "vertical ", "horizontal " })
        for (int i = 0; i < runtime_flags.get_number_of_shifts(); ++i) {
            //P
            double phase = (CV_2PI * i) / runtime_flags.get_number_of_shifts();

        }


}

cv::Mat ImageProcessing::mean(std::vector<cv::Mat> vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i].size() != vec[0].size() || vec[i].type() != vec[0].type()) {
            std::cerr << "All pictures must be same kind and type. \n";
            throw std::runtime_error ("All pictures must be same kind and type. \n");
        }
    }
    cv::Mat acc;
    vec[0].convertTo(acc, CV_32FC3);

    for (size_t i = 1; i < vec.size(); ++i) {
        cv::Mat temp;
        vec[i].convertTo(temp, CV_32FC3);
        acc += temp;   // pixelweise Addition
    }

    acc /= static_cast<float>(vec.size());  // pixelweise Division

    // Zurück zu 8 Bit
    cv::Mat average;
    acc.convertTo(average, CV_8UC3);
}
