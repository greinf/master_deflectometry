#ifndef SCENE_HPP
#define SCENE_HPP

#include "Utils.hpp"
#include "Light.hpp"
#include "Mesh.hpp"
#include "TraceAlbeMesh.hpp"
#include "Camera.hpp"
#include "BRDF.hpp"
#include "BVH.hpp"
#include <deque>
#include <memory>
#include <algorithm>
#include <functional>
#include <variant>
#include <thread>
#include <atomic>
#include <mutex>
#include <array>
#include <cmath>


using SceneObjPtr = std::variant<std::monostate, TriangularMesh*, Light*>;

class Scene{
public:
	Scene() = default;

	Scene(
		std::vector<std::unique_ptr<Camera>>&& cam,
		std::vector<std::unique_ptr<TriangularMesh>>&& mesh,
		std::vector<std::unique_ptr<Light>>&& light)
		: m_cam{std::move(cam)}
		, m_mesh{std::move(mesh)}
		, m_light{std::move(light)} { }

	void addLight(std::unique_ptr<Light>&& light) 
	{
		if (light == nullptr) throw std::invalid_argument("PTR was nullptr");
		m_light.push_back(std::move(light));
	}

	void addObject(std::unique_ptr<TriangularMesh>&& mesh)
	{
		if (mesh == nullptr) throw std::invalid_argument("PTR was nullptr");
		m_mesh.push_back(std::move(mesh));
	}

	void addCamera(std::unique_ptr<Camera>&& cam)
	{
		if (cam == nullptr) throw std::invalid_argument("PTR was nullptr");
		m_cam.push_back(std::move(cam));
	}
	
	void raytraceScene(std::vector<Eigen::MatrixXd>& out_img)
	{
		const std::size_t rays_per_thread{ 100 };

		if (m_cam.empty()) {
			throw std::runtime_error("No Camera in the scene");
			return;
		}
		if (m_mesh.empty()) std::cout << "No object in the scene \n";
		if (m_light.empty()) std::cout << "No Light sources in the scene \n";

		out_img.clear();
		out_img.reserve(m_cam.size());

		// Create the instance of ThreadPool
		ThreadPool& thread_p = ThreadPool::instance();
		// Create the future Object
		std::vector<std::shared_future<std::vector<double>>> result_task_All;
		
		// Work for every Camera seperatly
		// For each loop and for each camera the world coordiante System must be the camera system
		for (const auto& cam : m_cam) {
			// Transform all vertices 
			transform_all(cam->getTransform());

			Rays rays{};
			cam->generateRays(rays);

			// Deque to have some safety for pointers 
			//std::deque<std::unique_ptr<TraceAbleMesh>> traceableMeshes;

			for (const auto& mesh : m_mesh) {
				m_workingMeshes.emplace_back(std::make_unique<BVH>(mesh.get()));
			}
			for (const auto& light_mesh : m_light) {
				const auto& mesh_opt{ light_mesh->getMesh() };
				// This is optional since at some Point Spot light might be implemented which should not be tested 
				if (!mesh_opt.has_value()) continue;
				const auto& mesh{ mesh_opt.value() };
				m_workingMeshes.emplace_back(std::make_unique<BVH>(light_mesh.get()));
			}

			out_img.push_back(Eigen::MatrixXd());
			Eigen::MatrixXd& result = out_img.back();
			result.resize(
				rays.rows(),
				rays.cols());

			const std::size_t numberOfRays{
				static_cast<std::size_t>(rays.size())};

			const std::size_t outerDim{ numberOfRays / rays_per_thread };

			const Eigen::Vector3d start = Eigen::Vector3d::Zero();


			auto task = [
				function = &Scene::castRay,
				instance_ptr = this,
				rays_per_thread,
				&rays,
				start](const std::size_t start_index) -> std::vector<double>
				{
					std::vector<double> results;
					results.reserve(rays_per_thread);

					const std::size_t end_index =
						std::min(
							start_index + rays_per_thread,
							static_cast<std::size_t>(rays.size())
						);

					for (std::size_t i = start_index; i < end_index; ++i)
					{
						RayStructure ray{
							{&rays.data()[i]},
							{&start}
						};

						results.push_back(std::invoke(function, instance_ptr, ray));
					}
					return results;
				};

			thread_p.start();

			for (std::size_t i = 0; i < numberOfRays; i += rays_per_thread) {
				std::size_t outerDim_current{ i / rays_per_thread };
				std::size_t innerDim_current{ i % rays_per_thread };

				std::shared_future<std::vector<double>> result = 
					thread_p.queueTask(task, i);

				result_task_All.push_back(std::move(result));
			}
			
			thread_p.stop();

			{ // Extract
				std::size_t i = 0;

				while (i < result_task_All.size()) {

					if (result_task_All[i].valid() &&
						result_task_All[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready)
					{
						std::vector<double> task_res = result_task_All[i].get();

						std::size_t target_counter = i * rays_per_thread;

						std::copy(task_res.begin(), task_res.end(), result.data() + target_counter);

						++i;
					}
					else {
						// Future ist noch nicht bereit: Kurze Pause und im nächsten Durchlauf dasselbe i prüfen
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
					}
				}
			}

			return;
		}
	}

	// Cast Rays into the Scene. Ray and origin point are provided
	[[nodiscard]] double castRay(
		RayStructure& ray
		)
	{
		double closestT = std::numeric_limits<double>::infinity();

		double closestU{};
		double closestV{};
				
		const TraceAbleMesh* closestObj{ nullptr };
		const TraceAbleMesh::Triangle* closestTriangle{ nullptr };
		

		for (auto& meshRef : m_workingMeshes) {
			double t{};
			double u{};
			double v{};

			TraceAbleMesh::Triangle* triangle_ptr{ nullptr };

			if (!meshRef->intersect(
				*ray.origin,
				*ray.dir,
				triangle_ptr,
				t, u, v))
			{
				continue;
			}

			if (t >= closestT) continue;

			closestT = t;
			closestU = u;
			closestV = v;

			closestObj = meshRef.get();

			closestTriangle = triangle_ptr;
		}

		if (std::isinf(closestT)) return m_background;

		return trace(
			closestObj,
			closestTriangle,
			ray,
			closestU,
			closestV,
			closestT);
	}

private:
	std::vector<std::unique_ptr<Camera>> m_cam{};
	std::vector<std::unique_ptr<TriangularMesh>> m_mesh{};
	std::vector<std::unique_ptr<Light>> m_light{};

	// New  09.08
	std::vector<std::unique_ptr<TraceAbleMesh>> m_workingMeshes{ };


	std::vector<std::unique_ptr<Eigen::MatrixXd>> m_results{};
	double m_background{};
	

	void transform_all(const Eigen::Matrix4d& cam_transform) {
		for (auto& obj : m_mesh) {
			obj->applyTransform(cam_transform);
		}
		for (auto& obj : m_light){
			obj->applyTransform(cam_transform);
		}
	}
	

	[[nodiscard]] double trace(
		const TraceAbleMesh* closestObj,
		const TraceAbleMesh::Triangle* closestTri,
		RayStructure& ray,
		const double& u,
		const double& v,
		const double& t)
	{
		const std::complex<double> refractive{
			TriangularMesh::get_Barycentric_Interpolated_refractive_index(
				*closestTri->vertex[0].refractive,
				*closestTri->vertex[1].refractive,
				*closestTri->vertex[2].refractive,
				u,
				v)
		};

		const Eigen::Vector3d vert_normal{
			TriangularMesh::get_Barycentric_Interpolated_vertex_normal(
				*closestTri->vertex[0].vertex_normal,
				*closestTri->vertex[1].vertex_normal,
				*closestTri->vertex[2].vertex_normal,
				u,
				v
		) };

		double cos_theta{ vert_normal.dot(-*ray.dir) };

		Eigen::Vector3d newOrigin{ t * *ray.dir };

		// Hardcoded Display Image !!! 
		if (closestObj->m_info->emitter == true) {
			const Eigen::Vector2d texture_coord{ TriangularMesh::get_Texture_Coord(
				*closestTri->vertex[0].uv_Vertice,
				*closestTri->vertex[1].uv_Vertice,
				*closestTri->vertex[2].uv_Vertice,
				u,
				v)
			};

			auto optLight{ closestObj->getLight() };

			if (!optLight.has_value()) {
				std::cout << "Mesh that should contain Mesh has no value \n";
				return m_background;
			}

			auto light_ptr{ optLight.value() };

			if (light_ptr == nullptr) {
				std::cout << "Alarm should contain a value \n";
			}

			// BRDF characeteristic of Source
			const double scalingBRDF{ BRDF(1.0, refractive).get_Reflection(cos_theta) };
			// Local Brightness in [0...1] 
			const double light_Brightness{ light_ptr->get_local_Texture(texture_coord, 0) };
			// Return maxLightPower * Angle- & distance- Scaling * Brightness IF the light is modular 
			return light_ptr->m_info.m_power * scalingBRDF * light_Brightness;
		}

		Eigen::Vector3d reflected = reflect_ray(
			*ray.dir,
			vert_normal);
		
		ray.dir = &reflected;
		ray.origin = &newOrigin;
		
		if (closestObj->m_info->specular) {
			const double scaling_BRDF{ BRDF(1.0, refractive).get_Reflection(cos_theta) };
			return scaling_BRDF * castRay(
					ray
				);
		}

		// Here the same 
		if (closestObj->m_info->diffuse) {
			throw std::invalid_argument("We do not have a pipeline for diffuse objects \n");
		}
		return m_background;
	}

	static Eigen::Vector3d reflect_ray(
		const Eigen::Vector3d& dir,
		const Eigen::Vector3d& surf_norm) 
	{
		Eigen::Vector3d dir_n = dir.normalized();
		Eigen::Vector3d surf_n = surf_norm.normalized();

		return dir_n - 2 * dir_n.dot(surf_n) * surf_n;
	}
};


#endif // "SCENE_HPP"