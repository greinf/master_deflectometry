#ifndef SCENE_HPP
#define SCENE_HPP

#include "Utils.hpp"
#include "Light.hpp"
#include "Mesh.hpp"
#include "TraceAlbeMesh.hpp"
#include "Camera.hpp"
#include "BRDF.hpp"
#include "BVH.hpp"
#include "Integrator.hpp"
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

struct SceneConfig {
	std::size_t display_images{ 4 }; // Must throw error if not aligned with light class
	std::size_t max_intersections{ 10 };
	double min_scaling_factor{ 1e-3 };
	double background{ 0 };
};

class Scene {
public:
	enum Material {
		Specular,
		Emitter,
		Diffuse
	};

private:
	struct ProtocolNode {
		double scaling = 1.0;
		std::size_t depth = 0;
		int parent_index = -1;
		int left_child = -1;
		int right_child = -1;
	};

	class HitProtocol
	{
	private:
		std::vector<ProtocolNode> nodes{};

	public:

		int add_root()
		{
			if (!nodes.empty())
				throw std::logic_error(
					"HitProtocol already contains nodes"
				);

			nodes.push_back(ProtocolNode{
				1.0,
				0,
				-1,
				-1,
				-1
				});

			return 0;
		}


		int add_node(const int parent_idx, const double scaling)
		{
			if (parent_idx < 0 ||
				static_cast<std::size_t>(parent_idx) >= nodes.size())
			{
				throw std::out_of_range(
					"HitProtocol::add_node: invalid parent index"
				);
			}

			const std::size_t parent =
				static_cast<std::size_t>(parent_idx);

			if (nodes[parent].left_child != -1 &&
				nodes[parent].right_child != -1)
			{
				throw std::logic_error(
					"HitProtocol node has more than two children"
				);
			}

			const double accumulated_scaling =
				nodes[parent].scaling * scaling;

			const std::size_t node_depth =
				nodes[parent].depth + 1;

			const int new_idx =
				static_cast<int>(nodes.size());

			nodes.push_back({
				accumulated_scaling,
				node_depth,
				parent_idx,
				-1,
				-1
				});

			if (nodes[parent].left_child == -1)
				nodes[parent].left_child = new_idx;
			else
				nodes[parent].right_child = new_idx;

			return new_idx;
		}

		[[nodiscard]]
		bool should_abort(
			const int node_idx,
			const double scale_limit,
			const std::size_t depth_limit) const
		{
			const auto& node =
				nodes.at(static_cast<std::size_t>(node_idx));

			return
				node.depth > depth_limit ||
				node.scaling < scale_limit;
		}


		void clear()
		{
			nodes.clear();
		}
	};

	SceneConfig m_config{};

public:
	Scene(const SceneConfig& config)
		:m_config{ config } {
	}

	Scene() = default;

	Scene(
		std::vector<std::unique_ptr<Camera>>&& cam,
		std::vector<std::unique_ptr<TriangularMesh>>&& mesh,
		std::vector<std::unique_ptr<Light>>&& light)
		: m_cam{ std::move(cam) }
		, m_mesh{ std::move(mesh) }
		, m_light{ std::move(light) } {
	}

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

	// The main Controll function from which every task gets started 
	void raytraceScene(std::vector<Eigen::MatrixXd>& out_img)
	{
		const std::size_t rays_per_thread{ 500 };

		if (m_cam.empty()) {
			throw std::runtime_error("No Camera in the scene");
			return;
		}
		if (m_mesh.empty()) std::cout << "No object in the scene \n";
		if (m_light.empty()) std::cout << "No Light sources in the scene \n";

		out_img.clear();
		out_img.reserve(m_cam.size());

		// Get the instance of ThreadPool
		ThreadPool& thread_p = ThreadPool::instance();
		// Create the future Object
		std::vector<std::shared_future<std::vector<double>>> result_task_All;

		Integrator& integrator{ Integrator::instance() };

		// Work for every Camera seperatly
		// For each loop and for each camera the world coordiante System must be the camera system
		for (const auto& cam : m_cam) {
			// Transform all vertices 

			// Clear the old working Vectors!!!
			m_workingMeshes.clear();
			integrator.clear();

			transform_all(cam->getTransform());

			Rays rays{};
			cam->generateRays(rays);

			// --- Create TraceAbleMeshes from Objects and Lights --- 
			// TraceAbleMeshes is the Working Class for processing the only holds pointer to the Mesh Data
			// TraceAbleMesh is the Base Class of BVH. BVH is used to directly create a Bounding Hirachy of the Meshes 
			// Also Integrator is gets Ptrs to the newly created TraceAbleMeshes and Light instances. 
			for (const auto& mesh : m_mesh) {
				m_workingMeshes.emplace_back(std::make_unique<BVH>(mesh.get()));
			}
			for (const auto& light_mesh : m_light) {
				const auto& mesh_opt{ light_mesh->getMesh() };
				// This is optional since at some Point Spot light might be implemented which should not be tested 
				if (!mesh_opt.has_value()) {
					integrator.addLight(light_mesh.get(), nullptr);
					continue;
				}
				const auto& mesh{ mesh_opt.value() };
				m_workingMeshes.emplace_back(std::make_unique<BVH>(light_mesh.get()));
				integrator.addLight(light_mesh.get(), m_workingMeshes.back().get());
			}

			// Ad all Traceable Objects (as ptr) to the Integrator
			for (const auto& workers : m_workingMeshes) {
				integrator.addTraceAbleObjects(workers.get());
			}

			IntegratorSettings integ_set{};

			// Here Some options can be applied to the intergrator by assinging values to Integrator Settings. 
			integrator.check(std::move(integ_set));

			out_img.push_back(Eigen::MatrixXd());
			Eigen::MatrixXd& result = out_img.back();
			result.resize(
				rays.rows(),
				rays.cols());

			const std::size_t numberOfRays{
				static_cast<std::size_t>(rays.size()) };

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

					HitProtocol protocoll{};

					for (std::size_t i = start_index; i < end_index; ++i)
					{
						const int root_idx{ protocoll.add_root() };

						RayStructure ray{
							{&rays.data()[i]},
							{&start}
						};

						results.push_back(std::invoke(function, instance_ptr, ray, protocoll, root_idx));

						protocoll.clear();
					}
					return results;
				};

			thread_p.start();


			for (std::size_t i = 0; i < numberOfRays; i += rays_per_thread) {
				std::shared_future<std::vector<double>> result =
					thread_p.queueTask(task, i);

				result_task_All.push_back(std::move(result));
			}

			thread_p.stop();

			{ // Extract
				for (std::size_t i = 0; i < result_task_All.size(); ++i) {
					std::vector<double> task_res = result_task_All[i].get();

					const std::size_t target_counter = i * rays_per_thread;

					std::copy(task_res.begin(), task_res.end(), result.data() + target_counter);
				}
			}

			result_task_All.clear();
		}

		return;
	}

	// Cast Rays into the Scene. Ray and origin point are provided
	[[nodiscard]] double castRay(
		RayStructure& ray,
		HitProtocol& protocol,
		const int parent_index)
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

		if (std::isinf(closestT)) return m_config.background;

		return trace(
			closestObj,
			closestTriangle,
			ray,
			closestU,
			closestV,
			closestT,
			protocol,
			parent_index);
	}

private:
	std::vector<std::unique_ptr<Camera>> m_cam{};
	std::vector<std::unique_ptr<TriangularMesh>> m_mesh{};
	std::vector<std::unique_ptr<Light>> m_light{};

	std::vector<std::unique_ptr<TraceAbleMesh>> m_workingMeshes{ };


	void transform_all(const Eigen::Matrix4d& cam_transform) {
		for (auto& obj : m_mesh) {
			obj->applyTransform(cam_transform);
		}
		for (auto& obj : m_light) {
			obj->applyTransform(cam_transform);
		}
	}

	[[nodiscard]] double trace(
		const TraceAbleMesh* closestObj,
		const TraceAbleMesh::Triangle* closestTri,
		RayStructure& ray,
		const double& u,
		const double& v,
		const double& t,
		HitProtocol& protocol,
		const int parent_index)
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
				v)
		};

		const double cos_theta{ std::clamp(vert_normal.dot(-ray.dir->normalized()), 0.0, 1.0) };

		constexpr double ray_epsilon = 1e-8;

		Eigen::Vector3d hitPoint =
			*ray.origin + t * *ray.dir;

		Eigen::Vector3d newOrigin =
			hitPoint + ray_epsilon * vert_normal.normalized();

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

			auto light_ptr{ optLight.value() };

			// BRDF characeteristic of Source
			const double scalingBRDF{ BRDF(1.0, refractive).get_Reflection(cos_theta) };
			// Local Brightness in [0...1] 
			const double light_Brightness{ light_ptr->get_local_Texture(texture_coord, 0) };
			// Return maxLightPower * Angle- * Brightness IF the light is modular

			int new_idx = protocol.add_node(parent_index, scalingBRDF);

			if (protocol.should_abort(new_idx, m_config.min_scaling_factor, m_config.max_intersections)) {
				return 0.0;
			}

			return light_ptr->m_info.m_power * scalingBRDF * light_Brightness;
		}

		Eigen::Vector3d reflected = reflect_ray(
			*ray.dir,
			vert_normal);

		ray.dir = &reflected;
		ray.origin = &newOrigin;

		if (closestObj->m_info->specular)
		{
			const double scaling_BRDF{ BRDF(1.0, refractive).get_Reflection(cos_theta) };
			// In this case Fine, but also a bit dangerous since there is no stopping condition!!! 

			int new_idx = protocol.add_node(parent_index, scaling_BRDF);

			if (protocol.should_abort(new_idx, m_config.min_scaling_factor, m_config.max_intersections))
				return 0.0;

			return scaling_BRDF * castRay(
				ray,
				protocol,
				new_idx
			);
		}

		if (closestObj->m_info->diffuse)
		{
			Integrator& instance = Integrator::instance();

			const double scattered_scaling = closestObj->m_info->m_diffuse_settings.reflectivity_scattered;

			const double direct_light_scaling = closestObj->m_info->m_diffuse_settings.reflectivity_direct;

			int scattered_index = protocol.add_node(parent_index, scattered_scaling);

			double scattered{ protocol.should_abort(
				scattered_index,
				m_config.min_scaling_factor,
				m_config.max_intersections) ? (0.0) : instance.evaluate(newOrigin, closestTri, cos_theta) };

			int direct_index = protocol.add_node(parent_index, direct_light_scaling);

			double direct{ protocol.should_abort(
				direct_index,
				m_config.min_scaling_factor,
				m_config.max_intersections) ? (0.0) : castRay(ray, protocol, direct_index) };

			return scattered_scaling * scattered + direct_light_scaling * direct;

		}
		return 0.0;
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