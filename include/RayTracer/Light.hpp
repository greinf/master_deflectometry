#ifndef LIGHT_HPP	
#define LIGHT_HPP

#include <Eigen/dense>
#include <memory>
#include <vector>
#include "Mesh.hpp"
#include "Random.hpp"
#include <optional>


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

	virtual void applyTransform(const Eigen::Matrix4d&) noexcept = 0;

	virtual void addTransform(const Eigen::Matrix4d& trans) noexcept = 0;

	virtual const std::optional<TriangularMesh*> getMesh() noexcept = 0;

	virtual const std::optional<const TriangularMesh*> getMesh() const noexcept = 0;

	virtual const double get_local_Texture(
		const Eigen::Vector2d& texture_Coord,
		const std::size_t& index
		) const noexcept = 0;

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
		// Wrapper for the Mesh Transformation
		m_info.add_Transform(&m_mesh->m_transform);
		m_mesh->m_info.diffuse = false;
		m_mesh->m_info.specular = false;
		m_mesh->m_info.emitter = true;
	}

	~AreaLight() override = default;

	void applyTransform(const Eigen::Matrix4d& cam) noexcept override {
		m_mesh->applyTransform(cam);
	}

	void addTransform(const Eigen::Matrix4d& trans) noexcept override {
		if (!m_mesh->m_transform.isIdentity()) {
			std::cout << "Transform does contain data \n";
			return;
		}
		m_mesh->m_transform = trans;
	}

	const double get_local_Texture(
		const Eigen::Vector2d& texture_coor,
		const std::size_t& i) const noexcept override
	{
		if (m_Texture == nullptr) return 1.0;

		Eigen::Index image_x{ m_Texture->begin()->cols() };
		Eigen::Index image_y{ m_Texture->begin()->rows() };

		const Eigen::Vector2i texture_index = texture_coor.array().round().cast<int>();
		const Eigen::Index x_i = std::clamp(
			static_cast<Eigen::Index>(texture_index.x()), 
			static_cast<Eigen::Index>(0), 
			image_x);

		const Eigen::Index y_i = std::clamp(
			static_cast<Eigen::Index>(texture_index.y()),
			static_cast<Eigen::Index>(0),
			image_y
		);

		const double val = m_Texture->at(i)(image_x, image_y);

		return std::clamp(val, 0.0, 1.0);
	}

	const std::optional<TriangularMesh*> getMesh() noexcept override 
	{
		return std::optional<TriangularMesh*>(m_mesh.get());
	}

	const std::optional<const TriangularMesh*> getMesh() const noexcept override {
		return std::optional<const TriangularMesh*>(m_mesh.get());
	}

	Light_Info m_info{};
	
	void add_Texture(std::unique_ptr<std::vector<Eigen::MatrixXd>>&& texture) {
		if (texture == nullptr) throw std::invalid_argument("Used nullptr for Texture");
		if (texture->empty()) throw std::invalid_argument("Give emtpy Vector for Texture");

		const Eigen::Index rows{ texture->begin()->rows() };
		const Eigen::Index cols{ texture->begin()->cols() };

		if (rows == 0 || cols || 0) throw std::invalid_argument("Empty Matrix for Texture");

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
		const std::size_t& pixel_x,
		const std::size_t& pixel_y,
		const double& pixel_pitch,
		const double& wavelength,
		const std::size_t shifts)
		: AreaLight{ std::move(mesh) }
		, m_pixel_x{pixel_x}
		, m_pixel_y{pixel_y}
		, m_pixel_pitch{pixel_pitch}
		, m_wavelength{wavelength}
		, m_n_shifts{shifts}
	{ }

	double m_wavelength{};
	
	// Return a Value between 0 ... 1
	const double get_local_Texture(
		const Eigen::Vector2d& texture_coor,
		const std::size_t& i) const noexcept override
	{
		return get_Cos(
			texture_coor.x(),
			texture_coor.y(),
			m_wavelength,
			i,
			m_n_shifts);
	}

private:
	std::size_t m_pixel_x{};
	std::size_t m_pixel_y{};
	std::size_t m_n_shifts{};
	double m_pixel_pitch{};
	
	// Start with horizontal Phase, after that vertical Phase 
	// Wavelength in mm per period
	const double get_Cos(
		const double& u, 
		const double& v,
		const double& wavelength,
		const std::size_t current_step, 
		const std::size_t n_steps) const 
	{
		if (current_step * 2 > n_steps) 
			throw std::invalid_argument("More Shits are wanted than available");
		
		const double phase{ (current_step % n_steps) * wavelength/static_cast<double>(n_steps)};
		// Integer division: zero for Horizontal
		if (current_step / n_steps == 0) {
			return std::cos(u * wavelength + phase) * 0.5 + 0.5;
		}
		return std::cos(v * wavelength + phase) * 0.5 + 0.5;
	}
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