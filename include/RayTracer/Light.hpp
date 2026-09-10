#ifndef LIGHT_HPP	
#define LIGHT_HPP

#include <Eigen/dense>
#include <memory>
#include <vector>
#include "Mesh.hpp"
#include "Random.hpp"
#include <optional>
#include <algorithm>
#include <iostream>
#include <stdexcept>


struct Light_Sample {
	const Eigen::Vector3d* m_surface_normal;
	Eigen::Vector3d m_sample{};
};

struct Light_Info{
private:
	// The base
	Eigen::Matrix4d m_transform_Base{ Eigen::Matrix4d::Identity() };

public:
	double m_power{};
	
	Eigen::Matrix4d* m_transform{ &m_transform_Base };

	// Only for the case of Meshes
	void add_Transform(Eigen::Matrix4d* trans) noexcept {
		m_transform = trans;
	}
	//double (*radiation)(const double& angle) = nullptr;

};

class Light {
public:
	Light() = default;

	virtual ~Light() = default;

	virtual void applyTransform(const Eigen::Matrix4d&) = 0;

	virtual void addTransform(const Eigen::Matrix4d& trans) noexcept = 0;

	virtual const std::optional<TriangularMesh*> getMesh() noexcept = 0;

	virtual const std::optional<const TriangularMesh*> getMesh() const noexcept = 0;

	virtual bool is_Display() const = 0;

	virtual std::optional<std::size_t> n_display_shifts() const noexcept = 0;

	virtual double get_local_Texture(
		const Eigen::Vector2d& texture_Coord,
		const std::size_t& index
		) const = 0;

	Eigen::Matrix4d getTransform()
	{
		if (m_info.m_transform == nullptr) {
			std::cout << "Transformation Matrix is nullptr \n";
			return Eigen::Matrix4d::Zero();
		}
		return *m_info.m_transform;
	}

	Light_Info m_info{};
};	


class AreaLight : public Light{
public:
	explicit AreaLight(
		std::unique_ptr<TriangularMesh>&& mesh)
		: Light()
		, m_mesh{ std::move(mesh) }
	{
		if (m_mesh == nullptr)
			throw std::invalid_argument("AreaLight received nullptr mesh");

		// Wrapper for the Mesh Transformation
		m_info.add_Transform(&m_mesh->m_transform);
		m_mesh->m_info.diffuse = false;
		m_mesh->m_info.specular = false;
		m_mesh->m_info.emitter = true;
	}

	~AreaLight() override = default;

	bool is_Display() const override { return false; }

	std::optional<std::size_t> n_display_shifts() const noexcept override {
		return std::nullopt;
	}

	void applyTransform(const Eigen::Matrix4d& cam) override {
		m_mesh->applyTransform(cam);
	}

	void addTransform(const Eigen::Matrix4d& trans) noexcept override {
		if (!m_mesh->m_transform.isIdentity()) {
			std::cout << "Transform does contain data \n";
			return;
		}
		m_mesh->m_transform = trans;
	}

	double get_local_Texture(
		const Eigen::Vector2d& texture_coor,
		const std::size_t& i) const override
	{
		if (m_Texture == nullptr) return 1.0;

		Eigen::Index image_x{ m_Texture->begin()->cols() };
		Eigen::Index image_y{ m_Texture->begin()->rows() };

		const Eigen::Index x_i = std::clamp<Eigen::Index>(
			static_cast<Eigen::Index>(texture_coor.x()),
			0,
			image_x - 1
		);

		const Eigen::Index y_i = std::clamp<Eigen::Index>(
			static_cast<Eigen::Index>(texture_coor.y()),
			0,
			image_y - 1
		);

		// Static area-light textures remain unchanged during a display
		// phase-shift sequence. If exactly one texture is present, reuse it.
		const std::size_t texture_index =
			(m_Texture->size() == 1) ? 0 : i;

		if (texture_index >= m_Texture->size())
			throw std::out_of_range("AreaLight texture index out of range");

		const double val = m_Texture->at(texture_index)(y_i, x_i);
		return val;
	}

	const std::optional<TriangularMesh*> getMesh() noexcept override
	{
		return std::optional<TriangularMesh*>(m_mesh.get());
	}

	const std::optional<const TriangularMesh*> getMesh() const noexcept override {
		return std::optional<const TriangularMesh*>(m_mesh.get());
	}
	
	void add_Texture(std::unique_ptr<std::vector<Eigen::MatrixXd>>&& texture) {
		if (texture == nullptr) throw std::invalid_argument("Used nullptr for Texture");
		if (texture->empty()) throw std::invalid_argument("Give emtpy Vector for Texture");

		const Eigen::Index rows{ texture->begin()->rows() };
		const Eigen::Index cols{ texture->begin()->cols() };

		if (rows == 0 || cols == 0) throw std::invalid_argument("Empty Matrix for Texture");

		for (auto it = texture->begin(); it != texture->end(); ++it) {
			if (it->rows() != rows || it->cols() != cols) 
				throw std::invalid_argument("All Matrixes must be the same size");
			if (it->maxCoeff() > 1.0 || it->minCoeff() < 0.0) 
				throw std::invalid_argument("Each Matrix must have Coeff between 0.0 and 1.0");
		}
		m_Texture = std::move(texture);
		return;
	}

private:
	std::unique_ptr<TriangularMesh> m_mesh{ nullptr };
	
	std::unique_ptr<std::vector<Eigen::MatrixXd>> m_Texture{ nullptr };
};

class Display : public AreaLight {
public:
	explicit Display(
		std::unique_ptr<TriangularMesh>&& mesh,
		std::size_t pixel_x,
		std::size_t pixel_y,
		double pixel_pitch,
		double wavelength,
		std::size_t shifts)
		: AreaLight{ std::move(mesh) }
		, m_pixel_x{ pixel_x }
		, m_pixel_y{ pixel_y }
		, m_pixel_pitch{ pixel_pitch }
		, m_wavelength{ wavelength }
		, m_n_shifts{ shifts }
	{
		if (m_pixel_x == 0 || m_pixel_y == 0 || m_pixel_pitch <= 0.0) {
			throw std::invalid_argument(
				"Display dimensions and pixel pitch must be positive"
			);
		}

		if (m_wavelength <= 0.0) {
			throw std::invalid_argument(
				"Wavelength must be positive"
			);
		}

		if (m_n_shifts == 0) {
			throw std::invalid_argument(
				"Number of phase shifts must be positive"
			);
		}
	}



	bool is_Display() const override { return true; }

	std::optional<std::size_t> n_display_shifts() const noexcept override {
		return m_n_shifts;
	}

	double get_local_Texture(
		const Eigen::Vector2d& texture_coord,
		const std::size_t& pattern_index) const override
	{
		if (pattern_index >= 2 * m_n_shifts) {
			throw std::out_of_range(
				"Pattern index exceeds available patterns"
			);
		}

		constexpr double two_pi{
			M_PI * 2.0
		};

		const bool second_orientation{
			pattern_index >= m_n_shifts
		};

		const std::size_t shift_index{
			pattern_index % m_n_shifts
		};

		const double coordinate{
			second_orientation
				? texture_coord.x()
				: texture_coord.y()
		};

		const double spatial_phase{
			two_pi * coordinate / m_wavelength
		};

		const double phase_shift{
			two_pi
			* static_cast<double>(shift_index)
			/ static_cast<double>(m_n_shifts)
		};

		return 0.5 * (
			std::cos(spatial_phase + phase_shift) + 1.0
			);
	}

	[[nodiscard]]
	std::size_t patternCount() const noexcept
	{
		return 2 * m_n_shifts;
	}

private:
	std::size_t m_pixel_x{};
	std::size_t m_pixel_y{};
	std::size_t m_n_shifts{};

	double m_pixel_pitch{};
	double m_wavelength{};
};


//class SpotLight : public Light {
//public:
//	SpotLight()
//		:Light() 
//	{ }
//
//	~SpotLight() override = default;
//
//	Eigen::Vector3d sample() const noexcept {
//		return m_transform.block<3, 1>(0, 3);
//	}
//
//	void applyTransform(const Eigen::Matrix4d& cam) noexcept override
//	{
//		if (!m_transform.isIdentity()) {
//			Eigen::Vector4d homogenuos{};
//			homogenuos.head<3>() = m_info.dir;
//			homogenuos(3) = 1.0;
//			m_info.dir = (m_transform * homogenuos).head<3>();
//			m_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
//		}
//	}
//
//	Eigen::Matrix4d& getTransform() noexcept {
//		return m_transform;
//	}
//
//private:
//};

#endif //LIGHT_HPP