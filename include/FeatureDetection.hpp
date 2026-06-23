#ifndef	FEATURE_DETECTION_HPP
#define FEATURE_DETECTION_HPP
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

struct FeatureDetection_Config {
	int n_working_features{ 100 };
    int n_output_features{ 1 };

	bool visualize = true;

	bool validData() const {
		if (n_working_features <= 0) return false;
		
		return true;
	}
};

struct FeatureDetection_Data {
	std::vector<cv::Mat>* img_primaryCam = nullptr;
	cv::Mat* mask_primary = nullptr;
    cv::Mat* cam_Mat_prim = nullptr;
    cv::Mat* distCoeffs_prim = nullptr;
	std::vector<cv::Mat>* img_secondaryCam = nullptr;
	cv::Mat* mask_secondary = nullptr;
    cv::Mat* cam_Mat_secon = nullptr;
    cv::Mat* distCoeffs_secon = nullptr;
	
	cv::Mat* essentialMatrix = nullptr;


	bool validData() const{
		// Check Mask
		if (mask_primary == nullptr) return false;
		if ((*mask_primary).empty()) return false;
		if ((*mask_primary).type() != CV_8U) return false;
		cv::Size size_prim_camera{ mask_primary->size()};
		if (mask_secondary == nullptr) return false;
		if ((*mask_secondary).empty()) return false;
		if ((*mask_secondary).type() != CV_8U) return false;
		cv::Size size_secon_camera{ mask_secondary->size() };

		// Check Input Images 
		if (img_primaryCam == nullptr) return false;
		if (img_primaryCam->size() < 1) return false;
		cv::Size img_size{ (*img_primaryCam)[0].size() };
		if (!std::all_of((*img_primaryCam).begin(), (*img_primaryCam).end(),
			[&](const cv::Mat& img) -> bool {
				return (img.size() == size_prim_camera) &&
					(img.type() == CV_64F);
			}
		)) return false;

		if (img_secondaryCam == nullptr) return false;
		if (img_secondaryCam->size() < 1) return false;
		if (!std::all_of((*img_secondaryCam).begin(), (*img_secondaryCam).end(),
			[&](const cv::Mat& img) -> bool {
				return (img.size() == size_secon_camera) &&
					(img.type() == CV_64F);
			}
		)) return false;

        // Cam Mat DistCoeffs Prim
        if (cam_Mat_prim == nullptr) return false;
        if (cam_Mat_prim->empty()) return false;
        if (cam_Mat_prim->size() != cv::Size(3, 3)) return false;
        if (distCoeffs_prim == nullptr) return false;
        if (distCoeffs_prim->empty()) return false;
        // Cam Mat DistCoeffs Secon
        if (cam_Mat_secon == nullptr) return false;
        if (cam_Mat_secon->empty()) return false;
        if (cam_Mat_secon->size() != cv::Size(3, 3)) return false;
        if (distCoeffs_secon == nullptr) return false;
        if (distCoeffs_secon->empty()) return false;
		// Check Essential MAtrix
		if (essentialMatrix == nullptr) return false;
		if (essentialMatrix->empty()) return false;
		if (essentialMatrix->size() != cv::Size(3, 3)) return false;

		return true;
	}
};


struct FeatureDetection_Result {
    cv::Point2f ptPrim{};
    cv::Point2f ptSecon{};
    float descriptorDistance = 0.0f;
    double epipolarError = 0.0;
    double score = std::numeric_limits<double>::infinity();
};


class FeatureDetection {
public:
	FeatureDetection(const FeatureDetection_Config& config)
		:m_config{ config }
	{
        if (!m_config.validData()) throw std::invalid_argument("InvalidArgument in FeatureDetectionConfig");
	}

	FeatureDetection_Result computeMatchingPoints(const FeatureDetection_Data& data) {
		if (!data.validData()) throw std::invalid_argument("Invalid Data in FeatureDetecionData");
		
        cv::Mat primaryWorking = cv::Mat::zeros((*data.img_primaryCam)[0].size(), CV_64F), 
            secondaryWorking = cv::Mat::zeros((*data.img_secondaryCam)[0].size(), CV_64F);

        for (std::size_t i = 0; i < data.img_primaryCam->size(); ++i) {
            primaryWorking += (*data.img_primaryCam)[i];
        }
        primaryWorking /= data.img_primaryCam->size();
        
        for (std::size_t i = 0; i < data.img_secondaryCam->size(); ++i) {
            secondaryWorking += (*data.img_secondaryCam)[i];
        }
        secondaryWorking /= data.img_secondaryCam->size();

        // SIFT is usually better than ORB for this kind of precision test.
        auto detector = cv::SIFT::create(m_config.n_working_features);

        std::vector<cv::KeyPoint> kpPrimary, kpSecondary;
        cv::Mat descPrimary, descSecondary;

        detector->detectAndCompute(primaryWorking, data.mask_primary, kpPrimary, descPrimary);
        detector->detectAndCompute(secondaryWorking, data.mask_secondary, kpSecondary, descSecondary);

        if (descPrimary.empty() || descSecondary.empty()) throw std::runtime_error("NO Features detected");

        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);

        std::vector<std::vector<cv::DMatch>> knnMatches;
        // Find the 2 best matches 
        matcher->knnMatchImpl(descPrimary, descSecondary, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        goodMatches.reserve(knnMatches.size());

        // We try to extract the best feature to work further on it 
        for (const auto& m : knnMatches)
        {
            if (m.size() < 2)
                continue;

            // We only take the match if it is significantly better than second best one. 
            if (m[0].distance < 0.75 * m[1].distance)
                goodMatches.push_back(m[0]);
        }

        
        std::vector<cv::Point2f> ptsPrimary, ptsSecondary;
        ptsPrimary.reserve(goodMatches.size());
        ptsSecondary.reserve(goodMatches.size());

        for (const auto& m : goodMatches)
        {
            ptsPrimary.push_back(kpPrimary[m.queryIdx].pt);
            ptsSecondary.push_back(kpSecondary[m.trainIdx].pt);
        }

        std::vector<FeatureDetection_Result> result(goodMatches.size());
        
        for (int i = 0; i < static_cast<int>(goodMatches.size()); ++i)
        {
            const auto& m = goodMatches[i];

            cv::Point2f pPrim = kpPrimary[m.queryIdx].pt;
            cv::Point2f pSecon = kpSecondary[m.trainIdx].pt;

            double epiErr = sampsonErrorEssential(
                pPrim,
                pSecon,
                *data.essentialMatrix,
                *data.cam_Mat_prim,
                *data.distCoeffs_prim,
                *data.cam_Mat_secon,
                *data.distCoeffs_secon
            );

            // Combined score:
            // lower descriptor distance and lower epipolar error are better.
            double score = static_cast<double>(m.distance) + 100.0 * epiErr;

            result[i] = FeatureDetection_Result{
                    pPrim,
                    pSecon,
                    m.distance,
                    epiErr,
                    score
                };
        }

        std::sort(result.begin(), result.end(),
            [](const FeatureDetection_Result& res1,
                const FeatureDetection_Result& res2) -> bool
            {
                return (res1.score > res2.score);
            });

        if (m_config.visualize == true) {
            visualizeKeypointMatches(
                result,
                primaryWorking,
                secondaryWorking
            );
        }

        if (result.empty())
            throw std::runtime_error("No valid matches after ratio test");

        FeatureDetection_Result best = *result.begin();

        return best;
    }

private:
	const FeatureDetection_Config& m_config;

    void visualizeKeypointMatches(
        std::vector<FeatureDetection_Result>& results,
        const cv::Mat& img_prim,
        const cv::Mat& img_secon)
    {
        for (auto it = results.begin(); it != results.end(); )
        {
            cv::Mat primary_worker_gray = img_prim.clone();
            cv::Mat secondary_worker_gray = img_secon.clone();

            cv::Mat primary_worker_color;
            cv::Mat secondary_worker_color;

            cv::cvtColor(primary_worker_gray, primary_worker_color, cv::COLOR_GRAY2BGR);
            cv::cvtColor(secondary_worker_gray, secondary_worker_color, cv::COLOR_GRAY2BGR);

            cv::drawMarker(
                primary_worker_color,
                it->ptPrim,
                cv::Scalar(0, 0, 255),
                cv::MARKER_CROSS,
                20,
                2
            );

            cv::drawMarker(
                secondary_worker_color,
                it->ptSecon,
                cv::Scalar(0, 0, 255),
                cv::MARKER_CROSS,
                20,
                2
            );

            cv::imshow("Primary_Marker", primary_worker_color);
            cv::imshow("Secondary_Marker", secondary_worker_color);
            cv::waitKey(1);

            std::cout << "Keep this match? [y/N]: ";

            char answer{};
            std::cin >> answer;

            if (answer == 'y' || answer == 'Y')
            {
                ++it; // keep element
                return;
            }
            else
            {
                it = results.erase(it); // delete element, returns next valid iterator
            }
        }

        cv::destroyWindow("Primary_Marker");
        cv::destroyWindow("Secondary_Marker");
    }
    
    static double sampsonErrorEssential(
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const cv::Mat& E,
        const cv::Mat& K1,
        const cv::Mat& distCoeffs1,
        const cv::Mat& K2,
        const cv::Mat& distCoeffs2)
    {
        cv::Matx33d Ex;
        E.convertTo(Ex, CV_64F);

        std::vector<cv::Point2f> points1 = { p1 }, points1_undistorted;
        std::vector<cv::Point2f> points2 = { p2 }, points2_undistorted;

        cv::undistortPoints(points1, points1_undistorted, K1, distCoeffs1);
        cv::undistortPoints(points2, points2_undistorted, K2, distCoeffs2);

        cv::Vec3d x1n(points1_undistorted[0].x, points1_undistorted[0].y, 1.0);
        cv::Vec3d x2n(points2_undistorted[0].x, points2_undistorted[0].y, 1.0);

        cv::Vec3d Ex1 = Ex * x1n;
        cv::Vec3d Etx2 = Ex.t() * x2n;

        double x2tEx1 = x2n.dot(Ex1);

        double denom =
            Ex1[0] * Ex1[0] +
            Ex1[1] * Ex1[1] +
            Etx2[0] * Etx2[0] +
            Etx2[1] * Etx2[1];

        if (denom < 1e-20)
            return std::numeric_limits<double>::infinity();

        return (x2tEx1 * x2tEx1) / denom;
    }
};

#endif