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

	~Light() = default;

	virtual void applyTransform(const Eigen::Matrix4d&) noexcept = 0;

	virtual void addTransform(const Eigen::Matrix4d& trans) noexcept = 0;

	virtual const std::optional<const TriangularMesh&> getMesh() const noexcept = 0;

	virtual const std::optional<const double&> get_local_Intensity_scale(const double& u, const double& v) const noexcept = 0;

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

class AreaLight : public Light{
public:
	explicit AreaLight(
		std::unique_ptr<TriangularMesh>&& mesh)
		: Light()
		, m_mesh{ std::move(mesh) }
	{
		// Wrapper for the Mesh Transformation
		m_info.add_Transform(&m_mesh->m_transform);
	}

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

	const std::optional<const double&> get_local_Intensity_scale() {

	}

	const std::optional<const TriangularMesh&> getMesh() const noexcept override 
	{
		return std::optional<const TriangularMesh&>(*m_mesh);
	}

	Light_Info m_info{};
	std::unique_ptr<TriangularMesh> m_mesh{ nullptr };
};


#endif //LIGHT_HPP