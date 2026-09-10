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
		for (const auto& camera : m_cam)
			if (camera == nullptr) throw std::invalid_argument("Scene received nullptr camera");

		for (const auto& object : m_mesh) {
			if (object == nullptr) throw std::invalid_argument("Scene received nullptr mesh");
			if (object->m_info.diffuse) object->m_info.m_diffuse_settings.validate();
		}

		for (const auto& source : m_light)
			if (source == nullptr) throw std::invalid_argument("Scene received nullptr light");
	}

	void addLight(std::unique_ptr<Light>&& light)
	{
		if (light == nullptr) throw std::invalid_argument("PTR was nullptr");
		m_light.push_back(std::move(light));
	}

	void addObject(std::unique_ptr<TriangularMesh>&& mesh)
	{
		if (mesh == nullptr) throw std::invalid_argument("PTR was nullptr");
		if (mesh->m_info.diffuse) mesh->m_info.m_diffuse_settings.validate();
		m_mesh.push_back(std::move(mesh));
	}

	void addCamera(std::unique_ptr<Camera>&& cam)
	{
		if (cam == nullptr) throw std::invalid_argument("PTR was nullptr");
		m_cam.push_back(std::move(cam));
	}

	[[nodiscard]] bool hasDisplay() const noexcept
	{
		return std::any_of(
			m_light.begin(),
			m_light.end(),
			[](const auto& light) { return light != nullptr && light->is_Display(); }
		);
	}
	

	// Checkup if every dispaly is showing the same Scene
	[[nodiscard]] std::optional<std::size_t> displayShiftCount() const
	{
		std::optional<std::size_t> shiftCount{};

		for (const auto& light : m_light) {
			if (light == nullptr || !light->is_Display()) continue;

			const auto current{ light->n_display_shifts() };
			if (!current.has_value())
				throw std::logic_error("Display does not provide a phase-shift count");

			if (shiftCount.has_value() && shiftCount.value() != current.value())
				throw std::logic_error("Displays use different phase-shift counts");

			shiftCount = current;
		}

		return shiftCount;
	}

	[[nodiscard]] std::optional<std::size_t> displayPatternCount() const
	{
		const auto shifts{ displayShiftCount() };
		if (!shifts.has_value())
			return std::nullopt;
		return 2 * shifts.value();
	}

	// Renders one selected display pattern. For scenes without a Display,
	// pattern_index must remain zero. Output contains one image per camera.
	void raytraceScene(
		std::vector<Eigen::MatrixXd>& out_img,
		const std::size_t pattern_index = 0)
	{
		if (m_cam.empty())
			throw std::runtime_error("No Camera in the scene");

		validatePatternIndex(pattern_index);

		if (m_mesh.empty()) std::cout << "No object in the scene \n";
		if (m_light.empty()) std::cout << "No Light sources in the scene \n";

		out_img.clear();
		out_img.reserve(m_cam.size());

		Integrator& integrator{ Integrator::instance() };

		for (const auto& cam : m_cam) {
			Rays rays{};
			prepareCamera(*cam, rays, integrator);
			out_img.push_back(renderPreparedRays(rays, pattern_index));
		}
	}

	// Convenience function for a complete two-orientation phase-shift
	// sequence. The current geometry transform implementation is destructive,
	// therefore this method intentionally supports one camera per Scene.
	// Geometry/BVH setup is performed once; only the display pattern changes.
	void raytracePhaseShiftSequence(std::vector<Eigen::MatrixXd>& out_img)
	{
		if (m_cam.empty())
			throw std::runtime_error("No Camera in the scene");
		if (m_cam.size() != 1)
			throw std::logic_error(
				"raytracePhaseShiftSequence currently requires exactly one camera");

		const auto patternCount{ displayPatternCount() };
		if (!patternCount.has_value())
			throw std::logic_error("No Display in scene for phase-shift rendering");

		out_img.clear();
		out_img.reserve(patternCount.value());

		Integrator& integrator{ Integrator::instance() };
		Rays rays{};
		prepareCamera(*m_cam.front(), rays, integrator);

		for (std::size_t pattern = 0; pattern < patternCount.value(); ++pattern)
			out_img.push_back(renderPreparedRays(rays, pattern));
	}

	// Cast Rays into the Scene. Ray and origin point are provided
	[[nodiscard]] double castRay(
		RayStructure& ray,
		HitProtocol& protocol,
		const int parent_index,
		const std::size_t pattern_index)
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
			parent_index,
			pattern_index);
	}

private:
	std::vector<std::unique_ptr<Camera>> m_cam{};
	std::vector<std::unique_ptr<TriangularMesh>> m_mesh{};
	std::vector<std::unique_ptr<Light>> m_light{};

	std::vector<std::unique_ptr<TraceAbleMesh>> m_workingMeshes{ };


	void validatePatternIndex(const std::size_t pattern_index) const
	{
		const auto count{ displayPatternCount() };

		if (!count.has_value()) {
			if (pattern_index != 0)
				throw std::out_of_range("Pattern index requires a Display in the Scene");
			return;
		}

		if (pattern_index >= count.value())
			throw std::out_of_range("Display pattern index out of range");
	}

	void prepareCamera(
		const Camera& cam,
		Rays& rays,
		Integrator& integrator)
	{
		m_workingMeshes.clear();
		integrator.clear();

		transform_all(cam.getTransform());
		cam.generateRays(rays);

		for (const auto& mesh : m_mesh)
			m_workingMeshes.emplace_back(std::make_unique<BVH>(mesh.get()));

		for (const auto& light_mesh : m_light) {
			const auto mesh_opt{ light_mesh->getMesh() };

			if (!mesh_opt.has_value()) {
				integrator.addLight(light_mesh.get(), nullptr);
				continue;
			}

			m_workingMeshes.emplace_back(std::make_unique<BVH>(light_mesh.get()));
			integrator.addLight(light_mesh.get(), m_workingMeshes.back().get());
		}

		for (const auto& worker : m_workingMeshes)
			integrator.addTraceAbleObjects(worker.get());

		IntegratorSettings settings{};
		integrator.check(std::move(settings));
	}

	Eigen::MatrixXd renderPreparedRays(
		Rays& rays,
		const std::size_t pattern_index)
	{
		validatePatternIndex(pattern_index);

		constexpr std::size_t rays_per_thread{ 500 };

		Eigen::MatrixXd result{};
		result.resize(rays.rows(), rays.cols());

		const std::size_t numberOfRays{ static_cast<std::size_t>(rays.size()) };
		const Eigen::Vector3d start{ Eigen::Vector3d::Zero() };

		ThreadPool& thread_p{ ThreadPool::instance() };
		std::vector<std::shared_future<std::vector<double>>> futures{};
		futures.reserve((numberOfRays + rays_per_thread - 1) / rays_per_thread);

		auto task = [
			function = &Scene::castRay,
			instance_ptr = this,
			rays_per_thread,
			&rays,
			start,
			pattern_index](const std::size_t start_index) -> std::vector<double>
		{
			std::vector<double> results{};
			results.reserve(rays_per_thread);

			const std::size_t end_index =
				std::min(
					start_index + rays_per_thread,
					static_cast<std::size_t>(rays.size()));

			HitProtocol protocol{};

			for (std::size_t i = start_index; i < end_index; ++i) {
				const int root_idx{ protocol.add_root() };

				RayStructure ray{
					{ &rays.data()[i] },
					{ &start }
				};

				results.push_back(std::invoke(
					function,
					instance_ptr,
					ray,
					protocol,
					root_idx,
					pattern_index));

				protocol.clear();
			}

			return results;
		};

		thread_p.start();

		for (std::size_t i = 0; i < numberOfRays; i += rays_per_thread)
			futures.push_back(thread_p.queueTask(task, i));

		thread_p.stop();

		for (std::size_t i = 0; i < futures.size(); ++i) {
			std::vector<double> task_res{ futures[i].get() };
			const std::size_t target_counter{ i * rays_per_thread };
			std::copy(task_res.begin(), task_res.end(), result.data() + target_counter);
		}

		return result;
	}
	
	// This leaves big room for improvement. At this point all objects get destroyed - transformed 
	// and recreated as TraceAbleMesh. Just Leaf the Objects transform the starting rays. 
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
		const int parent_index,
		const std::size_t pattern_index)
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
 
		if (closestObj->m_info->emitter == true) {

			const Eigen::Vector2d texture_coord{
				TriangularMesh::get_Texture_Coord(
					*closestTri->vertex[0].uv_Vertice,
					*closestTri->vertex[1].uv_Vertice,
					*closestTri->vertex[2].uv_Vertice,
					u,
					v)
			};

			auto optLight{ closestObj->getLight() };
			auto light_ptr{ optLight.value() };

			const double light_Brightness{
				light_ptr->get_local_Texture(texture_coord, pattern_index)
			};

			return light_ptr->m_info.m_power * light_Brightness;
		}


		if (closestObj->m_info->specular)
		{
			Eigen::Vector3d reflected{
				reflect_ray(*ray.dir, vert_normal)
			};

			RayStructure reflectedRay{
				{ &reflected },
				{ &newOrigin }
			};

			Schlick_Approximation schlick{ 1.0, refractive };
			const double scaling_BRDF{
				schlick(cos_theta)
			};

			int new_idx = protocol.add_node(parent_index, scaling_BRDF);

			if (protocol.should_abort(
				new_idx,
				m_config.min_scaling_factor,
				m_config.max_intersections))
			{
				return 0.0;
			}

			return scaling_BRDF * castRay(
				reflectedRay,
				protocol,
				new_idx,
				pattern_index
			);
		}

		if (closestObj->m_info->diffuse)
		{
			Integrator& instance = Integrator::instance();

			const Eigen::Vector3d outgoing_dir{
				(-*ray.dir).normalized()
			};

			if (closestTri->diffuse_settings == nullptr)
				throw std::logic_error("Diffuse triangle has no material settings");

			return instance.evaluate(
				newOrigin,
				closestTri,
				vert_normal,
				outgoing_dir,
				*closestTri->diffuse_settings,
				pattern_index
			);
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