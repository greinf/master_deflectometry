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
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

struct IntegratorSettings {
    std::size_t n_samples_per_point{ 50 };
    std::unique_ptr<RandomGenerator> generator{ nullptr };
};

// Small Helper Class that slightly extends the TraceAbleMesh
class TraceAbleMeshIntegrator
{
public:
    TraceAbleMeshIntegrator(const TraceAbleMesh* mesh)
        : m_mesh{ mesh }
    {
        m_section.reserve(mesh->m_n_triangles);

        for (std::size_t i = 0; i < mesh->m_n_triangles; ++i) {
            m_section.push_back(
                *mesh->m_triangle[i].area / *mesh->m_area
            );
        }
    }

    std::vector<double> m_section{};
    const TraceAbleMesh* m_mesh{ nullptr };
};

class Integrator {

private:
    Integrator() = default;

    // Light sources
    std::vector<std::pair<Light*, std::optional<TraceAbleMeshIntegrator>>> m_lights{};

    // All TraceAbleObjects Including Light Sources
    std::vector<const TraceAbleMesh*> m_Objects{};

    IntegratorSettings m_settings{};

    static Eigen::Vector3d reflect_ray(
        const Eigen::Vector3d& dir,
        const Eigen::Vector3d& surf_norm)
    {
        Eigen::Vector3d dir_n = dir.normalized();
        Eigen::Vector3d surf_n = surf_norm.normalized();

        return dir_n - 2.0 * dir_n.dot(surf_n) * surf_n;
    }

    static double phong_specular_brdf(
        const Eigen::Vector3d& wi,
        const Eigen::Vector3d& wo,
        const Eigen::Vector3d& normal,
        const ObjectInfo::Diffuse_Settings& material)
    {
        if (material.reflectivity_direct <= 0.0)
            return 0.0;

        const Eigen::Vector3d incident{
            -wi.normalized()
        };

        const Eigen::Vector3d reflected{
            reflect_ray(incident, normal)
        };

        const double cos_alpha{
            std::max(0.0, reflected.dot(wo.normalized()))
        };

        if (cos_alpha <= 0.0)
            return 0.0;

        const double exponent{
            material.specular_exponent
        };

        const double normalization{
            (exponent + 2.0) / (2.0 * M_PI)
        };

        return material.reflectivity_direct
            * normalization
            * std::pow(cos_alpha, exponent);
    }

    double sampleLight(
        const TraceAbleMeshIntegrator* mesh,
        const Light* light_ptr,
        const Eigen::Vector3d& hitpoint,
        const TraceAbleMesh::Triangle* hit_tri,
        const Eigen::Vector3d& shading_normal,
        const Eigen::Vector3d& outgoing_dir,
        const ObjectInfo::Diffuse_Settings& material,
        const std::size_t pattern_index)
    {
        double intensity{ 0.0 };

        const double totalArea{ *mesh->m_mesh->m_area };
        const std::size_t totalSamples{ m_settings.n_samples_per_point };

        const Eigen::Vector3d surfaceNormal{
            shading_normal.normalized()
        };

        const Eigen::Vector3d wo{
            outgoing_dir.normalized()
        };

        if (surfaceNormal.dot(wo) <= 0.0) {
            return 0.0;
        }

        const double diffuse_brdf{
            material.reflectivity_scattered / M_PI
        };

        for (std::size_t i = 0; i < mesh->m_mesh->m_n_triangles; ++i)
        {
            const auto& lightTri = mesh->m_mesh->m_triangle[i];

            const double triangleArea{ *lightTri.area };

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
                    std::sqrt(static_cast<double>(barycentric.first))
                };

                const double e1{
                    sqrtE1 *
                    (1.0 - static_cast<double>(barycentric.second))
                };

                const double e2{
                    sqrtE1 *
                    static_cast<double>(barycentric.second)
                };

                const Eigen::Vector3d sample{
                    TriangularMesh::get_Coords_from_Barycentric(
                        *lightTri.vertex[0].pos,
                        *lightTri.vertex[1].pos,
                        *lightTri.vertex[2].pos,
                        e1,
                        e2
                    )
                };

                // vector from light sample -> surface point
                const Eigen::Vector3d toSurface{
                    hitpoint - sample
                };

                const double distanceSquared{
                    toSurface.squaredNorm()
                };

                if (distanceSquared <= 1e-24)
                    continue;

                const double distance{
                    std::sqrt(distanceSquared)
                };

                const Eigen::Vector3d dir{
                    toSurface / distance
                };

                // dir points light -> surface
                const double cosLight{
                    lightTri.surf_norm->dot(dir)
                };

                // wi points surface -> light
                const Eigen::Vector3d wi{
                    -dir
                };

                const double cosSurface{
                    surfaceNormal.dot(wi)
                };

                if (cosLight <= 0.0 || cosSurface <= 0.0)
                    continue;

                RayStructure shadowRay{
                    { &dir },
                    { &sample }
                };

                if (obscured(
                    shadowRay,
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
                        pattern_index
                    )
                };

                const double Le{
                    light_ptr->m_info.m_power *
                    lightTexture
                };

                const double specular_brdf{
                    phong_specular_brdf(
                        wi,
                        wo,
                        surfaceNormal,
                        material
                    )
                };

                const double brdf{
                    diffuse_brdf + specular_brdf
                };

                triangleContribution +=
                    Le *
                    brdf *
                    cosLight *
                    cosSurface /
                    distanceSquared;
            }

            intensity +=
                triangleArea *
                triangleContribution /
                static_cast<double>(samplesPerTri);
        }

        return intensity;
    }

    bool obscured(
        const RayStructure& ray,
        const TraceAbleMesh* light_ptr,
        const double& distance) const
    {
        const double eps{ 1e-6 * std::max(1.0, distance) };
        const double maxDistance{ distance - eps };

        if (maxDistance <= 0.0)
            return false;

        for (const auto& obj : m_Objects) {
            // The sampled light must not occlude itself.
            if (obj == light_ptr)
                continue;

            // Visibility is intentionally two-sided: opaque geometry blocks
            // a shadow ray independent of triangle winding/backface culling.
            if (obj->intersectAny(
                *ray.origin,
                *ray.dir,
                maxDistance))
            {
                return true;
            }
        }

        return false;
    }

    std::pair<float, float> getBarycentricSample() const
    {
        const float e1{ (*m_settings.generator)() };
        const float e2{ (*m_settings.generator)() };

        return { e1, e2 };
    }

public:
    static Integrator& instance() {
        static Integrator inst{};
        return inst;
    }

    void addLight(Light* light_ptr, TraceAbleMesh* traceAble = nullptr) {
        if (light_ptr == nullptr)
            throw std::invalid_argument("Integrator::addLight received nullptr");

        std::optional<TraceAbleMeshIntegrator> optionalTrace =
            (traceAble == nullptr)
            ? std::nullopt
            : std::optional<TraceAbleMeshIntegrator>(traceAble);

        m_lights.emplace_back(light_ptr, std::move(optionalTrace));
    }

    void clear() noexcept {
        m_lights.clear();
        m_Objects.clear();
    }

    void addTraceAbleObjects(const TraceAbleMesh* mesh) {
        if (mesh == nullptr)
            throw std::invalid_argument("Integrator::addTraceAbleObjects received nullptr");

        m_Objects.push_back(mesh);
    }

    void check(IntegratorSettings&& settings = {})
    {
        if (settings.n_samples_per_point == 0)
            throw std::invalid_argument("Integrator requires at least one sample per point");

        if (settings.generator == nullptr) {
            std::cout << "Switch to default Mersenne Twister \n";
            settings.generator = std::make_unique<MersenneTwister>();
        }

        if (m_lights.empty()) {
            std::cout << "WARNING: No Light instances in integrator \n";
        }

        if (m_Objects.empty()) {
            std::cout << "WARNING: No Objects to Test in Integrator Class \n";
        }

        for (const auto& light_src : m_lights) {
            if (!light_src.second.has_value())
                throw std::logic_error(
                    "Integrator currently supports only traceable area lights"
                );
        }

        m_settings = std::move(settings);
    }

    double evaluate(
        const Eigen::Vector3d& hitpoint,
        const TraceAbleMesh::Triangle* hit_tri,
        const Eigen::Vector3d& shading_normal,
        const Eigen::Vector3d& outgoing_dir,
        const ObjectInfo::Diffuse_Settings& material,
        const std::size_t pattern_index)
    {
        material.validate();

        double intensity{};

        for (const auto& light_src : m_lights) {
            if (!light_src.second.has_value()) {
                throw std::logic_error(
                    "Non-mesh light reached Integrator::evaluate"
                );
            }

            intensity += sampleLight(
                &light_src.second.value(),
                light_src.first,
                hitpoint,
                hit_tri,
                shading_normal,
                outgoing_dir,
                material,
                pattern_index
            );
        }

        return intensity;
    }
};

#endif // INTEGRATOR_HPP