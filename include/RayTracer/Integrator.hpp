#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include "Light.hpp"
#include <Eigen/dense>
#include <vector>


class Integrator {
	static double sampleLights(
		const std::size_t n_points_perLight,
		const Light* src,
		const std::vector<TriangularMesh*> object,
		RandomGenerator* rand)
	{
		for (const auto& light_src : src) {

		}
	}

};





#endif //INTEGRATOR_HPP