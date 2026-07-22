#ifndef SCENE_HPP
#define SCENE_HPP

#include "Light.hpp"
#include "Mesh.hpp"
#include "Camera.hpp"
#include "BRDF.hpp"
#include "BVH.hpp"
#include <vector>
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

// Traceable Mesh builds a wrapper around the 
class TraceAbleMesh {
public:
	TraceAbleMesh(TriangularMesh* mesh)
		:m_mesh{ mesh } 
	{
		build();
	}
	TraceAbleMesh(Light* light)
		:m_light{ light }
	{
		auto opt = light->getMesh();
		if (opt.has_value()) m_mesh = opt.value();
		else throw std::runtime_error("Light is not traceable");
		build();
	}
	
	std::size_t m_n_triangles{};
	
	std::optional<const Light*> getLight() { return std::optional<const Light*>(m_light); }
	
	virtual bool intersect(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const Eigen::Vector3d& v0,
		const Eigen::Vector3d& v1,
		const Eigen::Vector3d& v2,
		double& t, double& u, double& v) const
	{
		return TriangularMesh::intersect(origin, dir, v0, v1, v2, t, u, v);
	}

	ObjectInfo* m_info{nullptr};

	std::optional<Light*> getLight() const {
		if (m_light == nullptr) return std::nullopt;
		return std::optional<Light*>(m_light);
	}

private:
	TriangularMesh* m_mesh{ nullptr };
	Light* m_light{ nullptr };

	struct Vertex {
		Eigen::Vector3d* pos{ nullptr };
		Eigen::Vector3d* vertex_normal{ nullptr };
		Eigen::Vector2d* uv_Vertice{ nullptr };
		std::complex<double>* refractive{ nullptr };
	};

	void build() {
		auto n_surfaces{ static_cast<std::size_t>(m_mesh->m_n_surfaces) };

		if (n_surfaces == 0) throw std::invalid_argument("Empty Mesh given");

		m_triangle = std::unique_ptr<Triangle[]>(new Triangle[n_surfaces]);

		m_n_triangles = n_surfaces;

		if (m_mesh->m_area == nullptr) {
			m_mesh->get_Area();
		}

		if (m_mesh->m_surface_normals == nullptr) {
			m_mesh->calc_surface_normals();
		}

		for (std::size_t i = 0; i < static_cast<std::size_t>(n_surfaces); ++i) {
			std::size_t index[3]{
				static_cast<std::size_t>(m_mesh->m_indices[i * 3]),
				static_cast<std::size_t>(m_mesh->m_indices[i * 3 + 1]),
				static_cast<std::size_t>(m_mesh->m_indices[i * 3 + 2])
			};
			// Check for indexing!!! 
			if (index[0] >= m_mesh->m_n_vertices ||
				index[1] >= m_mesh->m_n_vertices ||
				index[2] >= m_mesh->m_n_vertices)
				throw std::out_of_range("You Failed misserably \n");

			m_triangle[i].area = &m_mesh->m_area[i];
			m_triangle[i].surf_norm = &m_mesh->m_surface_normals[i];
			m_triangle[i].vertex[0].pos = &m_mesh->m_vertices[index[0]].pos;
			m_triangle[i].vertex[1].pos = &m_mesh->m_vertices[index[1]].pos;
			m_triangle[i].vertex[2].pos = &m_mesh->m_vertices[index[2]].pos;
			m_triangle[i].vertex[0].refractive = &m_mesh->m_vertices[index[0]].refractive_index;
			m_triangle[i].vertex[1].refractive = &m_mesh->m_vertices[index[1]].refractive_index;
			m_triangle[i].vertex[2].refractive = &m_mesh->m_vertices[index[2]].refractive_index;
			m_triangle[i].vertex[0].uv_Vertice = &m_mesh->m_vertices[index[0]].uv;
			m_triangle[i].vertex[1].uv_Vertice = &m_mesh->m_vertices[index[1]].uv;
			m_triangle[i].vertex[2].uv_Vertice = &m_mesh->m_vertices[index[2]].uv;
			m_triangle[i].vertex[0].vertex_normal = &m_mesh->m_vertice_normals[index[0]];
			m_triangle[i].vertex[1].vertex_normal = &m_mesh->m_vertice_normals[index[1]];
			m_triangle[i].vertex[2].vertex_normal = &m_mesh->m_vertice_normals[index[2]];
			
			m_info = &m_mesh->m_info;
		}
	}
public: 
	struct Triangle {
		Eigen::Vector3d* surf_norm{ nullptr };
		Vertex vertex[3]{};
		double* area{};
	};

	std::unique_ptr<Triangle[]> m_triangle{ nullptr };
};


class Scene{
private: 
	struct RayStructure {
		const Eigen::Vector3d* dir;
		const Eigen::Vector3d* origin;
	};

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
		if (m_cam.empty()) {
			throw std::runtime_error("No Camera in the scene");
			return;
		}
		if (m_mesh.empty()) std::cout << "No object in the scene \n";
		if (m_light.empty()) std::cout << "No Light sources in the scene \n";

		out_img.clear();
		out_img.reserve(m_cam.size());

		// Work for every Camera seperatly
		// For each loop and for each camera the world coordiante System must be the camera system
		for (const auto& cam : m_cam) {
			// Transform all vertices 
			transform_all(cam->getTransform());
			
			Rays rays{};
			cam->generateRays(rays);

			std::vector<TraceAbleMesh> traceableMeshes{};

			for (const auto& mesh : m_mesh) {
				traceableMeshes.push_back(TraceAbleMesh(mesh.get()));
			}
			for (const auto& light_mesh : m_light) {
				const auto& mesh_opt{ light_mesh->getMesh() };
				// This is optional since at some Point Spot light might be implemented which should not be tested 
				if (!mesh_opt.has_value()) continue;
				const auto& mesh{ mesh_opt.value() };
				traceableMeshes.push_back(TraceAbleMesh(light_mesh.get()));
			}
			
			out_img.push_back(Eigen::MatrixXd());
			Eigen::MatrixXd& result = out_img.back();
			result.resize(
				rays.rows(),
				rays.cols());

			const std::size_t numberOfRays{
				static_cast<std::size_t>(rays.size())};

			const std::size_t numberOfWorkers{
				getWorkerCount(numberOfRays)};

			if (numberOfWorkers == 0) {
				continue;
			}

			const std::size_t raysPerWorker{
				(numberOfRays + numberOfWorkers - 1)
				/ numberOfWorkers
			};

			std::vector<std::thread> workers{};
			workers.reserve(numberOfWorkers);

			// Exceptions must not escape directly from a std::thread.
			// Otherwise std::terminate() is called.
			std::exception_ptr workerException{};
			std::mutex exceptionMutex{};
			std::atomic_bool stopRequested{ false };

			Eigen::Vector3d start = Eigen::Vector3d::Zero();

			try {
				for (std::size_t workerIndex = 0;
					workerIndex < numberOfWorkers;
					++workerIndex)
				{
					const std::size_t begin{
						workerIndex * raysPerWorker
					};

					const std::size_t end{
						std::min(
							begin + raysPerWorker,
							numberOfRays
						)
					};

					if (begin >= end) {
						break;
					}

					workers.emplace_back(
						[&, begin, end]()
						{
							try {
								for (std::size_t rayIndex = begin;
									rayIndex < end;
									++rayIndex)
								{
									RayStructure ray{
										{&rays.data()[rayIndex]},
										{&start}
									};

									if (stopRequested.load(
										std::memory_order_relaxed))
									{
										return;
									}

									result.data()[rayIndex] = castRay(
										ray,
										traceableMeshes,
										m_background
									);
								}
							}
							catch (...) {
								stopRequested.store(
									true,
									std::memory_order_relaxed
								);

								std::lock_guard<std::mutex> lock{
									exceptionMutex
								};

								if (!workerException) {
									workerException =
										std::current_exception();
								}
							}
						}
					);
				}
			}
			catch (...) {
				stopRequested.store(
					true,
					std::memory_order_relaxed
				);

				for (std::thread& worker : workers) {
					if (worker.joinable()) {
						worker.join();
					}
				}

				throw;
			}

			for (std::thread& worker : workers) {
				worker.join();
			}

			if (workerException) {
				std::rethrow_exception(workerException);
			}
		}
	}
	// Cast Rays into the Scene. Ray and origin point are provided
	[[nodiscard]] double castRay(
		RayStructure& ray,
		const std::vector<TraceAbleMesh>& everyMesh,
		const double& background)
	{
		double closestT = std::numeric_limits<double>::infinity();

		double closestU{};
		double closestV{};
				
		const TraceAbleMesh* closestObj{ nullptr };
		const TraceAbleMesh::Triangle* closestTriangle{ nullptr };

		for (const auto& meshRef : everyMesh) {

			for (std::size_t i = 0;
				i < meshRef.m_n_triangles;
				++i)
			{
				double t{};
				double u{};
				double v{};

				if (!meshRef.intersect(
					*ray.origin,
					*ray.dir,
					*meshRef.m_triangle[i].vertex[0].pos,
					*meshRef.m_triangle[i].vertex[1].pos,
					*meshRef.m_triangle[i].vertex[2].pos,
					t, u, v))
				{
					continue;
				}

				if (t >= closestT) continue;

				closestT = t;
				closestU = u;
				closestV = v;

				closestObj = &meshRef;

				closestTriangle = &meshRef.m_triangle[i];
			}
		}

		if (std::isinf(closestT)) return m_background;

		return trace(
			everyMesh,
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
	
	[[nodiscard]] static std::size_t getWorkerCount(
		const std::size_t numberOfTasks) noexcept
	{
		if (numberOfTasks == 0) {
			return 0;
		}

		const unsigned int hardwareHint{
			std::thread::hardware_concurrency()
		};

		// may return 0 if not possibible to determine
		const std::size_t availableThreads{
			hardwareHint == 0
				? std::size_t{1}
				: static_cast<std::size_t>(hardwareHint)
		};

		// Never create more workers than there are tasks.
		return std::min(availableThreads, numberOfTasks);
	}

	[[nodiscard]] double trace(
		const std::vector<TraceAbleMesh>& meshes,
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
					ray,
					meshes,
					m_background
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