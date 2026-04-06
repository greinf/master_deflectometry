#include "ImageStore.hpp"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

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


void ImageStore::add(FrameRole role, const std::array<std::pair<double, double>, 256>& LUT) {
    CV_Assert(role == FrameRole::GrayLUT);
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

std::array<std::pair<double, double>,256> ImageStore::getLut() const {
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
    if (role == FrameRole::GrayLUT) {
        //if (!LUT_Gray.empty()) return false;
        return !std::all_of(LUT_Gray.begin(), LUT_Gray.end(),
            [&](const std::pair<double, double>& value) -> bool
            {
                return value.first == 0 && value.second == 0;
            });
    }

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
    const std::string& name = FrameRole_string[static_cast<std::size_t>(role)];
    
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
            cv::imshow(name, uchar);
            u = cv::waitKey(0);
            cv::destroyWindow(name);
            if (std::toupper(u) == 'Q') break;
        }
    }
    else std::cout << "Not a valid Key \n";
}

std::vector<cv::Mat>* ImageStore::getPtr(const FrameRole role)
{
    std::scoped_lock lock(mtx_);
    CV_Assert(has(role));
    auto it = storage_.find(role);

    return &(it->second);
}


void ImageStore::saveLut(const std::string& basePath)
{
    if (LUT_Gray.empty()) {
        std::cout << "Data is empty\n";
        return;
    }

    if (std::all_of(LUT_Gray.begin(), LUT_Gray.end(),
        [&](const std::pair<double, double>& gray_val)
        {
            return gray_val.first == 0 && gray_val.second == 0;
        })) 
    {
        std::cout << "Array was not assigend, every entry is 0!. Return to caller \n";
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

std::pair<std::vector<cv::Mat>::const_iterator, std::vector<cv::Mat>::const_iterator>
ImageStore::getIter(FrameRole role) const 
{
    // GrayLut is not stored as a container.
    CV_Assert(role != FrameRole::GrayLUT);
    
    std::lock_guard<std::mutex> lock(mtx_);

    std::pair<std::vector<cv::Mat>::const_iterator, std::vector<cv::Mat>::const_iterator> iters{};

    auto it = storage_.find(role);
    if (it == storage_.end())
    {
        std::cerr << "Frame role not created \n";
        return {};
    }
    iters.first = it->second.begin();
    iters.second = it->second.end();
    return iters;
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

    std::array<std::pair<double, double>, 256> loaded;

    std::string line;
    int count{};
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);

        double x, y;
        char comma;

        ss >> x >> comma >> y;

        if (!ss.fail() && comma == ',')
            loaded[count++] = { x, y };
        else
            throw std::runtime_error("Malformed CSV line: " + line);
    }

    LUT_Gray = loaded;
}

void ImageStore::loadCalibrationMatrix(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (fs.isOpened()) {
        std::vector<cv::Mat> distCoeffs(2);
        std::vector<cv::Mat> camMatrix(2);
        {
            cv::FileNode distortion(fs["distortion_coefficients"]);
            if (distortion.empty()) {
                std::cout << "Loading Camera Data from StereoCalibration \n";

                fs["distortion_coefficients1"] >> distCoeffs.at(0);
                fs["distortion_coefficients2"] >> distCoeffs.at(1);
            }
            else distortion >> distCoeffs.at(0);
            
            cv::FileNode camMatrixn(fs["camera_matrix"]);
            if (camMatrixn.empty()) {
                fs["camera_matrix1"] >> camMatrix.at(0);
                fs["camera_matrix2"] >> camMatrix.at(1);
            }

            for (const auto& mat : camMatrix) {
                if (mat.empty()) continue;
                storage_[FrameRole::CalibrationMatrix].push_back(mat);
            }

            for (const auto& mat : distCoeffs) {
                if (mat.empty()) continue;
                storage_[FrameRole::DistortionCoeff].push_back(mat);
            }
        }
    }
    else std::cout << "Coudl not open the Storage file ";
    fs.release();
}

void ImageStore::loadCalibCamToCam(const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx_);
    storage_[FrameRole::CalibCamToCam].clear();
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Could not open XML file: " + path);

    cv::Mat rotationMat, translationMat;

    fs["rotationMat"] >> rotationMat;
    fs["translationMat"] >> translationMat;

    storage_[FrameRole::CalibCamToCam].push_back(rotationMat);
    storage_[FrameRole::CalibCamToCam].push_back(translationMat);

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

    

    if (role == FrameRole::CalibCamToCam) {
        loadCalibCamToCam(basePath);
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

