#ifndef LIGHT_HPP	
#define LIGHT_HPP

#include <Eigen/dense>
#include <memory>
#include <vector>
#include "Mesh.hpp"
#include "Random.hpp"


struct Light_Sample {
	const Eigen::Vector3d* m_surface_normal;
	Eigen::Vector3d m_sample{};
	
};



class Light: public Object{
public:
	Light()
		:Object() 
	{ }

	~Light() override = default;

	virtual void applyTransform(const Eigen::Matrix4d&) noexcept = 0;

	void addTransform(const Eigen::Matrix4d& trans) noexcept {
		m_transform = trans;
	}
	
	virtual Eigen::Matrix4d& getTransform() noexcept = 0;
};	


class SpotLight : public Light {
public:
	SpotLight()
		:Light() 
	{ }

	~SpotLight() override = default;

	Eigen::Vector3d sample() const noexcept {
	
	}

	void applyTransform(const Eigen::Matrix4d& cam) noexcept override
	{
		if (!m_transform.isIdentity()) {
			Eigen::Vector4d homogenuos{};
			homogenuos.head<3>() = m_info.dir;
			homogenuos(3) = 1.0;
			m_info.dir = (m_transform * homogenuos).head<3>();
			m_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
		}
	}

	Eigen::Matrix4d& getTransform() noexcept {
		return m_transform;
	}

private:
	struct SpotLightInfo {
		Eigen::Vector3d pos{};
		Eigen::Vector3d dir{};
	};

	SpotLightInfo m_info{};
};

class AreaLightSpot : Public SpotLight {

};



class AreaLight : public Light {
public:
	explicit AreaLight(
		std::unique_ptr<TriangularMesh>&& mesh)
		: Light()
		, m_mesh{ std::move(mesh) }
	{ }

	void applyTransform(const Eigen::Matrix4d& cam) noexcept override {
		m_mesh->applyTransform(cam);
	}

	Eigen::Matrix4d& getTransform() noexcept override {
		return m_mesh->m_transform;
	}

	//Eigen::Vector3d sample() const noexcept override
	//{
	//	float sample[2];
	//	// Sample two uniform values
	//	{
	//		sample[0] = MersenneTwister::generate();
	//		sample[1] = MersenneTwister::generate();
	//	}

	//	float uv[2];
	//	const float sqrt_s0{ std::sqrt(sample[0]) };
	//	uv[0] = sqrt_s0 * (1 - sample[1]);
	//	uv[1] = sqrt_s0 * sample[1];
	//}

private:
	std::unique_ptr<TriangularMesh> m_mesh{ nullptr };
};



#endif //LIGHT_HPP