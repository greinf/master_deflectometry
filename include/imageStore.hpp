#ifndef IMAGESTORE_H
#define IMAGESTORE_H

#include <map>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <array>
#include <string>


// What type of image is stored?
enum class FrameRole {
    Pattern,           // Pattern generated in Pattern()
    PatternDouble,     // Pattern in doulbe generated for Testing of Wrapp/Unwrapp
    GrayCalibrationGT, // Frames for Gray value Calibration generate in Pattern()
    GrayCalibrationCam,// Frames acquierd with camera for calibration
    GrayLUT,           // LUT Table for GrayValue calib
    RawPhase,          // Raw Images of Pattern (mean over all pics per Phase)
    RawInput,          // Raw images from camera, All images seperate
    WrappedPhase,      // Wrapped phase images
    Contrast,          // Contrast Wrapped Phase
    BaseIntensity,     // Base Intensity Wrapped Phase
    UnwrappedPhase,    // Unwrapped phase images
    Calibration,       // Images of Checkerboard for calibration
    CalibrationMatrix, // Calibration Matrix 
    DistortionCoeff,   // Distortion Coeffs
    CalibImages,       // Images of Checkerboard with found Points marked for controll
    ReprojectionX,     // Image that contains the X-component of the errorvector, from the reprojection on the display
    ReprojectionY,     // Image that contains the Y-component of the errorvector, from the reprojection on the display
    Debug,             // For any debug intermediate outputs
    all,               // Command for saving and loading
    GridPattern,       // GridPattern -> A grid of crosses over the hole image 
    DistortionCalib,   // A 2 Channel picture where the value at each pixel represent the original coordinates before the distortion. 
    DistortErrX,       // Picture that showcases the ammount of Distortion in the X component
    DistortErrY,       // Picture that showcases the ammount of Distortion in the Y component
    UndistortErrX,     // Picture that showscases the ammount of Distortion after the undistortion in the X compoment.
    UndistortErrY,      // Picture that showcases the ammound of Distortion after the undistortion in the Y compoment. 
    maxElements        // Place Holder for ammound of categories
};


class ImageStore {
public:
    ImageStore() = default;

    // Add image to store under a specific role
    void add(FrameRole role, const cv::Mat& img);
    void add(FrameRole role, const std::vector<std::pair<double, double>>&&);

    // Access images by category
    std::vector<cv::Mat> get(FrameRole role) const;
    std::vector<std::pair<double, double>> getLut() const;

    // Only get last image of category
    cv::Mat getLast(FrameRole role) const;

    // Delete images of a category
    void clear(FrameRole role);

    // Delete absolutely everything
    void clearAll();

    // (Optional) Query if images exist
    bool has(FrameRole role) const;

    // Number of stored images for role
    size_t count(FrameRole role) const;

    // Shows the hole vector of image category
    void show(FrameRole role) const;

    //void moveRawInputTo(FrameRole role);


    // I/O Functionality of Image Store.
    void saveRole(FrameRole role, const std::string& path);

    void saveRoleXML(FrameRole role, const std::string& filename);

    // load Role allows to read in from file the according data. The data must be in strucutre of
    // |-Path
    // |----|
    // |----|-Role1
    // |----|-Role2
    // |----...
    void loadRoleXML(FrameRole role, const std::string& filename);

    void loadCalibrationMatrix(const std::string& filename);

private:
    std::map<FrameRole, std::vector<cv::Mat>> storage_;
    std::vector<std::pair<double, double>> LUT_Gray;
    mutable std::mutex mtx_;  

    void saveLut(const std::string& path);
    void loadLut(const std::string& path);

    inline static const std::array<std::string,
        static_cast<size_t>(FrameRole::maxElements)> FrameRole_string{ {
        "Pattern", "PatternDouble", "GrayCalibrationGT", "GrayCalibrationCam", "GrayLUT",
        "RawPhase", "RawInput", "WrappedPhase","Contrast", 
        "BaseIntensity", "UnwrappedPhase","CalibrationImg", "CalibrationMatrix", 
        "DistortionCoefficients", "CalibrationImagesMarked", "ReprojectionX",
        "ReprojectionY", "Debug", "All", "GridPattern", "DistortionCalib", "DistortErrX",
        "DistortErrY", "UndistortErrX", "UndistortErrY"
    } };
    
};

#endif