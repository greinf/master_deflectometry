#include "ImageStore.hpp"
#include <filesystem>
#include <fstream>

//std::scoped_lock lock(m1, m2, m3); can lock multple mutex objects



void ImageStore::add(FrameRole role, const cv::Mat& img) {
    std::scoped_lock lock(mtx_); // the scoped lock can be extended
    storage_[role].push_back(img.clone());   // ensure deep copy
}

void ImageStore::add(
    FrameRole role,
    const std::vector<cv::Mat>& images)
{
    if (has(role) && !images.empty()) {
        std::cout << "WARNING: The Category aleady holds images. \n" <<
            "The images will be appended \n";
    }
    for (const auto& img : images) {
        add(role, img);
    }
}


void ImageStore::add(FrameRole role, const std::vector<std::pair<double, double>>&& LUT) {
    std::scoped_lock lock(mtx_);
    LUT_Gray = LUT;
}

std::vector<cv::Mat> ImageStore::get(FrameRole role) const {
    std::scoped_lock lock(mtx_);

    if (role == FrameRole::GrayLUT) {
        std::cout << "Call getLut() for gettig Lut \n";
        return {};
    }

    auto it = storage_.find(role);
    if (it != storage_.end())
        return it->second;

    return {};
}

std::vector<std::pair<double, double>> ImageStore::getLut() const {
    return LUT_Gray;
}

cv::Mat ImageStore::getLast(FrameRole role) const {
    std::scoped_lock lock(mtx_);

    auto it = storage_.find(role);
    if (it != storage_.end() && !it->second.empty())
        return it->second.back();

    return {};   // empty matrix
}

void ImageStore::clear(FrameRole role) {
    std::scoped_lock lock(mtx_);
    storage_[role].clear();
}

void ImageStore::clearAll() {
    std::scoped_lock lock(mtx_);
    storage_.clear();
}

bool ImageStore::has(FrameRole role) const {
    std::scoped_lock lock(mtx_);
    auto it = storage_.find(role);
    return (it != storage_.end() && !it->second.empty());
}

size_t ImageStore::count(FrameRole role) const {
    std::scoped_lock lock(mtx_);
    auto it = storage_.find(role);
    if (it != storage_.end())
        return it->second.size();
    return 0;
}

void ImageStore::show(FrameRole role) const {
    std::scoped_lock lock(mtx_);
    auto it = storage_.find(role);
    if (it != storage_.end()) {
        std::cout << "Valid Key. \n" <<
            "Press Button for iterating \n" <<
            "Press 'q' for breaking \n";
        for (const cv::Mat& image : it->second) {
            char u;
            cv::Mat uchar;
            if (image.type() != CV_8U) {
                cv::normalize(image, uchar, 0, 255, cv::NORM_MINMAX, CV_8U);
            }
            else uchar = image;
            cv::imshow("Map", uchar);
            u = cv::waitKey(0);
            cv::destroyWindow("Map");
            if (std::toupper(u) == 'Q') break;
        }
    }
    else std::cout << "Not a valid Key \n";
}


void ImageStore::saveLut(const std::string& basePath)
{
    if (LUT_Gray.empty()) {
        std::cout << "Data is empty\n";
        return;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    std::string dir = basePath + "/" +
        FrameRole_string.at(static_cast<size_t>(FrameRole::GrayLUT));

    std::filesystem::create_directories(dir);

    std::ofstream file(dir + "/lut.csv");
    if (!file.is_open())
        throw std::runtime_error("Could not open LUT file for writing");

    for (const auto& [x, y] : LUT_Gray)
        file << x << "," << y << "\n";
    file.close();

}


void ImageStore::saveRole(FrameRole role, const std::string& path)
{
    if (role == FrameRole::GrayLUT) {
        saveLut(path);
        return;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    auto it = storage_.find(role);
    if (it == storage_.end() || it->second.empty()) {
        std::cerr << "No images for this role.\n";
        return;
    }

    std::string Role_path = { path + "/" + FrameRole_string.at(static_cast<std::size_t>(role)) };
    std::filesystem::create_directories(Role_path);

    int idx = 0;
    for (const auto& img : it->second) {
        std::string filename = Role_path + "/" + std::to_string(idx++) + ".png";
        cv::Mat Mat8U;
        if (img.type() != CV_8U) cv::normalize(img, Mat8U, 0, 255, cv::NORM_MINMAX, CV_8U);
        else Mat8U = img;
        cv::imwrite(filename, Mat8U);
    }
}


void ImageStore::saveRoleXML(FrameRole role, const std::string& basePath)
{
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = storage_.find(role);
    if (it == storage_.end() || it->second.empty()) {
        std::cerr << "No images for this role.\n";
        return;
    }

    std::string dir = basePath + "/" +
        FrameRole_string.at(static_cast<size_t>(role));

    std::filesystem::create_directories(dir);

    cv::FileStorage fs(dir + "/data.xml", cv::FileStorage::WRITE);
    fs << "count" << (int)it->second.size();

    for (size_t i = 0; i < it->second.size(); i++)
        fs << ("img_" + std::to_string(i)) << it->second[i];
}


void ImageStore::loadLut(const std::string& basePath)
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::string dir = basePath + "/" +
        FrameRole_string.at(static_cast<size_t>(FrameRole::GrayLUT));

    std::string filePath = dir + "/lut.csv";

    if (!std::filesystem::exists(filePath)) {
        std::cout << "LUT not found: " << filePath << "\n";
        return;
    }

    std::ifstream file(filePath);
    if (!file.is_open())
        throw std::runtime_error("Cannot open LUT file");

    std::vector<std::pair<double, double>> loaded;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);

        double x, y;
        char comma;

        ss >> x >> comma >> y;

        if (!ss.fail() && comma == ',')
            loaded.emplace_back(x, y);
        else
            throw std::runtime_error("Malformed CSV line: " + line);
    }

    LUT_Gray = std::move(loaded);
}

void ImageStore::loadCalibrationMatrix(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (fs.isOpened()) {
        cv::Mat cam_mat;
        fs["distortion_coefficients"] >> cam_mat;
        storage_[FrameRole::DistortionCoeff].push_back(cam_mat);
        cv::Mat dist_coeffs;
        fs["camera_matrix"] >> dist_coeffs;
        storage_[FrameRole::CalibrationMatrix].push_back(dist_coeffs);
    }
    else std::cout << "Coudl not open the Storage file ";
    fs.release();
}

void ImageStore::loadRoleXML(FrameRole role, const std::string& basePath)
{
    if (role == FrameRole::GrayLUT) {
        loadLut(basePath);
        return;
    }

    if (role == FrameRole::CalibrationMatrix) {
        loadCalibrationMatrix(basePath);
        return;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    storage_[role].clear();

    std::string dir = basePath + "/" +
        FrameRole_string.at(static_cast<size_t>(role));

    std::string filePath = dir + "/data.xml";

    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Could not open XML file: " + filePath);

    int count;
    fs["count"] >> count;

    for (int i = 0; i < count; i++) {
        cv::Mat img;
        fs["img_" + std::to_string(i)] >> img;
        storage_[role].push_back(img);
    }
}

