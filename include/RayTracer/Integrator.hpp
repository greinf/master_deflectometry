#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include "Light.hpp"
#include <Eigen/dense>
#include <vector>


class Integrator {
	explicit Integrator(

	)




	double sampleLights(
		const std::size_t n_points_perLight,
		const AreaLight* src,
		const std::vector<TriangularMesh*> object,
		RandomGenerator* rand)
	{
		const std::uint32_t n_surface{ src->m_mesh->m_n_surfaces };
		const double area{ src->m_mesh->get_Area() };

		std::vector<float> area(n_surface);

		for (std::uint32_t i = 0; i < n_surface; ++i) {
			const Eigen::Vector3d{src->m_}
			area[i] = 1/2 * ()
		}

	}

};





#endif //INTEGRATOR_HPP