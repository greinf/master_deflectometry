#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include "Utils.hpp"
#include "TraceAlbeMesh.hpp"
#include "Light.hpp"
#include "Object.hpp"
#include <Eigen/dense>
#include <vector>
#include <optional>
#include <memory>

struct IntegratorSettings {
	std::size_t n_samples_per_point{ 50 };
	std::unique_ptr<RandomGenerator> generator{ nullptr };
};

// Small Helper Class that slightly extends the TraceAbleMesh 
class TraceAbleMeshIntegrator
{
public:
	TraceAbleMeshIntegrator(const TraceAbleMesh* mesh)
		:m_mesh{mesh}
		{
			m_section.reserve(mesh->m_n_triangles);
			for (std::size_t i = 0; i < mesh->m_n_triangles; ++i) {
				m_section.push_back(*mesh->m_triangle[i].area / *mesh->m_area);
			}
		}
	
	std::vector<double> m_section{};

	const TraceAbleMesh* m_mesh{ nullptr };
};

class Integrator {

private:
	Integrator() = default;

	// Light sources
	std::vector<std::pair<Light*, std::optional<TraceAbleMeshIntegrator>>> m_lights{ };

	// All TraceAbleObjects Including Light Sources
	std::vector<const TraceAbleMesh*> m_Objects{};

	IntegratorSettings m_settings{ };

    double sampleLight(
        const TraceAbleMeshIntegrator* mesh,
        const Light* light_ptr,
        const Eigen::Vector3d& hitpoint,
        const TraceAbleMesh::Triangle* hit_tri)
    {
        double intensity{ 0.0 };

        const double totalArea{ *mesh->m_mesh->m_area };
        const std::size_t totalSamples{ m_settings.n_samples_per_point };

        for (std::size_t i = 0;
            i < mesh->m_mesh->m_n_triangles;
            ++i)
        {
            const auto& lightTri = mesh->m_mesh->m_triangle[i];

            const double triangleArea{ *lightTri.area };

            // Desired number of samples proportional to triangle area.
            const std::size_t samplesPerTri{
                std::max<std::size_t>(
                    1,
                    static_cast<std::size_t>(
                        std::round(
                            totalSamples * triangleArea / totalArea
                        )
                    )
                )
            };

            double triangleContribution{ 0.0 };

            for (std::size_t s = 0; s < samplesPerTri; ++s)
            {
                const auto barycentric = getBarycentricSample();

                const double sqrtE1{
                    std::sqrt(
                        static_cast<double>(barycentric.first)
                    )
                };

                const double e1{
                    sqrtE1 *
                    (1.0 - static_cast<double>(barycentric.second))
                };

                const double e2{
                    sqrtE1 *
                    static_cast<double>(barycentric.second)
                };

                // Uniform sample on triangle
                const Eigen::Vector3d sample{
                    TriangularMesh::get_Coords_from_Barycentric(
                        *lightTri.vertex[0].pos,
                        *lightTri.vertex[1].pos,
                        *lightTri.vertex[2].pos,
                        e1,
                        e2
                    )
                };

                // Vector from light sample -> surface point
                const Eigen::Vector3d toSurface{
                    hitpoint - sample
                };

                const double distanceSquared{
                    toSurface.squaredNorm()
                };

                const double distance{
                    std::sqrt(distanceSquared)
                };

                if (distance <= 1e-12)
                    continue;

                const Eigen::Vector3d dir{
                    toSurface / distance
                };

                // cos(theta_light)
                const double cosLight{
                    lightTri.surf_norm->dot(dir)
                };

                // Direction from surface -> light is -dir
                const double cosSurface{
                    hit_tri->surf_norm->dot(-dir)
                };

                if (cosLight <= 0.0 || cosSurface <= 0.0)
                    continue;

                RayStructure shadowRay{
                    { &dir },
                    { &sample }
                };

                if (obscured(
                    shadowRay,
                    hit_tri,
                    mesh->m_mesh,
                    distance))
                {
                    continue;
                }

                const Eigen::Vector2d uvCoords{
                    TriangularMesh::get_Texture_Coord(
                        *lightTri.vertex[0].uv_Vertice,
                        *lightTri.vertex[1].uv_Vertice,
                        *lightTri.vertex[2].uv_Vertice,
                        e1,
                        e2
                    )
                };

                const double lightTexture{
                    light_ptr->get_local_Texture(
                        uvCoords,
                        0
                    )
                };

                const double Le{
                    light_ptr->m_info.m_power *
                    lightTexture
                };

                triangleContribution +=
                    Le *
                    cosLight *
                    cosSurface /
                    distanceSquared;
            }

            // Monte-Carlo integral over THIS triangle
            intensity +=
                triangleArea *
                triangleContribution /
                static_cast<double>(samplesPerTri);
        }

        // Lambert BRDF
        return intensity / M_PI;
    }

	// Function to check if possible light ray is obscured by Meshes included in the scene
	bool obscured(
		const RayStructure& ray,
		const TraceAbleMesh::Triangle* hit_tri,
		const TraceAbleMesh* light_ptr,
		const double& distance) const
	{
		// Variables for checking if first hit is achieved on specific Triangle
		TraceAbleMesh::Triangle* triangle_ptr{ nullptr };

		double t_{}, u_{}, v_{};

		for (const auto& obj : m_Objects) {
			// We do excpect that light sources can not obscure itself
			if (obj == light_ptr) continue;

			if (!obj->intersect(
				*ray.origin,
				*ray.dir,
				triangle_ptr,
				t_, u_, v_)
				) continue;
			else {
                const double eps = 1e-6 * std::max(1.0, distance);

                if (t_ < distance - eps) {
                    return true;
                }
			}
		}
		return false;
	}


	std::pair<float, float> getBarycentricSample() const
	{
		//std::cout << m_settings.generator

		float e1{ m_settings.generator->operator()() };
		float e2{ m_settings.generator->operator()() };

		return { e1, e2 };
	}

public:
	static Integrator& instance() {
		static Integrator inst{};
		return inst;
	}

	void addLight(Light* light_ptr, TraceAbleMesh* traceAble = nullptr) noexcept {

		std::optional<TraceAbleMeshIntegrator> optionalTrace = (traceAble == nullptr) ?
			(std::nullopt) : (std::optional<TraceAbleMeshIntegrator>(traceAble));

		m_lights.push_back(
			std::pair<Light*, std::optional<TraceAbleMeshIntegrator>>{light_ptr, optionalTrace});

	}

	void clear() noexcept {
		m_lights.clear();
		m_Objects.clear();
	}

	void addTraceAbleObjects(const TraceAbleMesh* mesh) noexcept{
		m_Objects.push_back(mesh);
	}

	// Integrator gets a seperator prepare class. This is done that every Thread that is opened
	// with it´s own instance of Merseene Twister, which can than be easiliy reused. 
	void check(IntegratorSettings&& settings = {}) 
	{
		if (m_lights.empty()) {
			std::cout << "WARNING: No Light instances in integrator \n";
			return;
		}
		if (settings.generator == nullptr) {
			std::cout << "Switch to default Mersenne Twister \n";
			settings.generator = std::make_unique<MersenneTwister>();
		}
		if (m_Objects.empty()) {
			std::cout << "WARNING: No Objects to Test in Integrator Class \n";
		}
		m_settings = std::move(settings);
	}

	double evaluate(
		//const BRDF* brdf,
		const Eigen::Vector3d& hitpoint,
		const TraceAbleMesh::Triangle* hit_tri,
		const double& cos_theta // cos Theta of angle between the incoming ray and surface
		)
	{
		double intensity{};

		for (const auto& light_src : m_lights) {
			// Spot Light 
			if (!light_src.second.has_value()) 
			{
				std::cout << "Not impelmented path. At this point no spotlights \n";
				return std::numeric_limits<double>::signaling_NaN();
			}

			// Area Light
			intensity += sampleLight(&light_src.second.value(), light_src.first, hitpoint, hit_tri);
		}


		return intensity;
	}
};


#endif //INTEGRATOR_HPP