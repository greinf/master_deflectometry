#ifndef GRAYCALIBVECTOR_HPP
#define GRAYCALIBVECTOR_HPP

#include <utils.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>


// All datapoints are stored as (x,y)
// This struct is used only for the creation of Response curve on dispaly.
// The display can be cut in Sections for that, so each section will get calculated seperatly. 
struct GrayCalibVector {
	std::vector<Gray_section_data> m_sections_img;

	void push_back(const Gray_section_data& data) {
		m_sections_img.push_back(data);
		check_img_borders(data);
	}

	void emplace_back(const Gray_section_data& data) {
		push_back(data);
	}

	void reserve(const std::size_t i) {
		m_sections_img.reserve(i);
	}

	void printHomography() {
		if (m_homographyMatrix.empty()) std::cout << "Homography not created \n";
		else std::cout << "HomographyMatrix: \n" << m_homographyMatrix << '\n';
	}

	std::vector<Gray_section_data>::iterator begin() {
		return m_sections_img.begin();
	}

	std::vector<Gray_section_data>::iterator end() {
		return m_sections_img.end();
	}

	std::vector<std::pair<int, int>> getFullBorders_img() {
		if (img_minX == img_minY == img_maxX == img_maxY) {
			std::cout << "Warning: the Borders are not set for the Image \n";
			return {};
		}
		std::vector<std::pair<int, int>> vec{};

		vec.push_back(std::pair<int, int>(img_minX, img_minY));
		vec.push_back(std::pair<int, int>(img_maxX, img_minY));
		vec.push_back(std::pair<int, int>(img_minX, img_maxY));
		vec.push_back(std::pair<int, int>(img_maxX, img_maxY));

		return vec;
	}


	// Saves the Complete Calib Response Data for all created section in a CSV file
	void save(const std::string& path) {
		if (m_sections_img.empty()) {
			std::cout << "Warning: save() called with empty m_sections_img\n";
			return;
		}

		// Basic consistency check: all sections should have same curve length
		const std::size_t L = m_sections_img.front().gray_val.size();
		if (L == 0) {
			std::cout << "Warning: save() called but gray_val is empty\n";
			return;
		}

		for (const auto& s : m_sections_img) {
			if (s.gray_val.size() != L || s.s_deviation.size() != L) {
				throw std::runtime_error("save(): inconsistent curve lengths across sections");
			}
		}

		std::ofstream os(path);
		if (!os) {
			throw std::runtime_error("save(): could not open file: " + path);
		}

		// CSV header (tidy format)
		os << "section_id,idx,gray_in,"
			"mean,std,"
			"img_x0,img_y0,img_x1,img_y1,"
			"obj_x0,obj_y0,obj_x1,obj_y1\n";

		os << std::setprecision(17);

		// If you want gray_in to represent actual input gray value:
		// We don't know n_steps here. Best effort:
		// - if L == 256: assume step=1
		// - else assume it spans [0..255] evenly: gray_in = round(idx * 255/(L-1))
		auto gray_in_from_idx = [L](std::size_t idx) -> int {
			if (L <= 1) return 0;
			if (L == 256) return static_cast<int>(idx);
			// scale index to [0,255]
			double g = (255.0 * static_cast<double>(idx)) / static_cast<double>(L - 1);
			return static_cast<int>(std::lround(g));
			};

		for (std::size_t sid = 0; sid < m_sections_img.size(); ++sid) {
			const auto& sec = m_sections_img[sid];

			const int img_x0 = sec.roi_img.left_up_corner.first;
			const int img_y0 = sec.roi_img.left_up_corner.second;
			const int img_x1 = sec.roi_img.right_up_corner.first;   // exclusive
			const int img_y1 = sec.roi_img.left_down_corner.second; // exclusive

			const double obj_x0 = sec.roi_obj.left_up_corner.first;
			const double obj_y0 = sec.roi_obj.left_up_corner.second;
			const double obj_x1 = sec.roi_obj.right_up_corner.first;
			const double obj_y1 = sec.roi_obj.left_down_corner.second;

			for (std::size_t i = 0; i < L; ++i) {
				const int gray_in = gray_in_from_idx(i);
				os << sid << ','
					<< i << ','
					<< gray_in << ','
					<< sec.gray_val[i] << ','
					<< sec.s_deviation[i] << ','
					<< img_x0 << ',' << img_y0 << ',' << img_x1 << ',' << img_y1 << ','
					<< obj_x0 << ',' << obj_y0 << ',' << obj_x1 << ',' << obj_y1
					<< '\n';
			}
		}
	}


private:
	int img_minX{}, img_minY{}, img_maxX{}, img_maxY{};

	//RoiBorders m_full_ROI_img_coordinate{};
	void check_img_borders(const Gray_section_data& a) {
		img_minX = std::min(a.roi_img.left_up_corner.first, img_minX);
		img_minY = std::min(a.roi_img.left_up_corner.second, img_minY);
		img_maxX = std::max(a.roi_img.right_down_corner.first - 1, img_maxX); // Be carefull the max values are exlusive
		img_maxY = std::max(a.roi_img.right_down_corner.second - 1, img_maxY);
		fullImgBorders.left_up_corner = std::pair<int, int>(img_minX, img_minY);
		fullImgBorders.left_down_corner = std::pair<int, int>(img_minX, img_maxY);
		fullImgBorders.right_up_corner = std::pair<int, int>(img_maxX, img_minY);
		fullImgBorders.right_down_corner = std::pair<int, int>(img_maxX, img_maxY);
	}
	cv::Mat m_homographyMatrix{};
	RoiBorders<int> fullImgBorders{};

	void sort_imgPts() {
		if (m_sections_img.empty()) {
			std::cout << "Warning: Tried to sort empty Array. \n";
			return;
		}

		std::sort(m_sections_img.begin(), m_sections_img.end(),
			[](const Gray_section_data& a, const Gray_section_data& b) {
				if (a.roi_img.left_up_corner.first != b.roi_img.left_up_corner.first)
					return a.roi_img.left_up_corner.first < b.roi_img.left_up_corner.first;   // primary: left → right
				return a.roi_img.left_up_corner.second < b.roi_img.left_up_corner.second;                   // secondary: top → bottom
			});
	}
};

#endif