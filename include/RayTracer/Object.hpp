#ifndef RAYTRACER_OBJECT_BASE_HPP
#define RAYTRACER_OJBECT_BASE_HPP	
#include <Eigen/dense>
#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdint>


#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2

class Object{
public:
	Object() = default;
	Object(const Eigen::Matrix4d& pos)
		:m_transform{pos}{ }

	void addTransform(const Eigen::Matrix4d& pos) noexcept {
		m_transform = pos;
	}

	virtual ~Object() = default;

	Eigen::Matrix4d m_transform = Eigen::Matrix4d::Identity();
};

// Utilities 
struct Vertice {
	Eigen::Vector3d pos{};
	double refractive_index{};

	bool specular{ false };

};





#endif // 'RAYTRACER_OJBECT_BASE_HPP