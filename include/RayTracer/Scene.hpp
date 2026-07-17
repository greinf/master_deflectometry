#ifndef SCENE_HPP
#define SCENE_HPP

#include "Light.hpp"
#include "Mesh.hpp"
#include "Camera.hpp"
#include "BRDF.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <variant>


class Scene {
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
	
	void raytraceScene(
		std::vector<Eigen::MatrixXd>& out_img)
	{
		if (m_cam.empty()) {
			throw std::runtime_error("No Camera in the scene");
			return;
		}
		if (m_mesh.empty()) std::cout << "No object in the scene \n";
		if (m_light.empty()) std::cout << "No Light sources in the scene \n";

		m_results.reserve(m_cam.size());
		
		for (auto& obj : m_mesh) {
			if (obj->m_surface_normals == nullptr) obj->calc_surface_normals();
		}

		for (auto& obj : m_light) {
			auto mesh_opt = obj->getMesh();
			if (mesh_opt.has_value()) {
				TriangularMesh* mesh{ mesh_opt.value() };
				if (mesh->m_surface_normals == nullptr) {
					mesh->calc_surface_normals();
				}
			}
		}

		// Work for every Camera seperatly
		// For each loop and for each camera the world coordiante System must be the camera system
		for (const auto& cam : m_cam) {
			// Transform all vertices 
			transform_all(cam->getTransform());
			
			Rays rays{};
			cam->generateRays(rays);
			
			m_results.push_back(std::make_unique<Eigen::MatrixXd>());
			std::unique_ptr<Eigen::MatrixXd>& result = m_results.back();
			result->resize(
				rays.rows(),
				rays.cols());

			for (Eigen::Index i = 0; i < rays.size(); ++i) {
				// Here threads could be created 
				result->data()[i] = castRay(
					Eigen::Vector3d::Zero(),
					rays.data()[i],
					m_mesh,
					m_light,
					m_background);
			}
		}
	}
	// Cast Rays into the Scene. Ray and origin point are provided
	double castRay(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const std::vector<std::unique_ptr<TriangularMesh>>& meshes,
		const std::vector<std::unique_ptr<Light>>& lights,
		const double& background)
	{
		double closestT = std::numeric_limits<double>::infinity();

		std::vector<std::pair<SceneObjPtr, std::reference_wrapper<const TriangularMesh>>> everyMesh{};
		// It would be possible to implement accelerating structures here.
		for (const auto& mesh : meshes) {
			everyMesh.push_back(
				std::pair<SceneObjPtr, std::reference_wrapper<const TriangularMesh>>(mesh.get(), *mesh));
		}
		for (const auto& light_mesh : lights) {
			const auto& mesh_opt{ light_mesh->getMesh() };
			// This is optional since at some Point Spot light might be implemented which should not be tested 
			if (!mesh_opt.has_value()) continue;
			const auto& mesh{ mesh_opt.value() };
			everyMesh.push_back(
				std::pair<SceneObjPtr, std::reference_wrapper<const TriangularMesh>>(light_mesh.get(), *mesh));
		}

		double closestU{};
		double closestV{};

		std::size_t closestIndex[3];
		std::size_t surfaceIndex{};
		
		SceneObjPtr closestObj{};
		
		for (const auto& meshRef : everyMesh) {
			const TriangularMesh& mesh = meshRef.second.get();

			for (std::size_t i = 0;
				i < static_cast<std::size_t>(mesh.m_n_surfaces);
				++i)
			{
				const std::size_t index0 = mesh.m_indices[i * 3];
				const std::size_t index1 = mesh.m_indices[i * 3 + 1];
				const std::size_t index2 = mesh.m_indices[i * 3 + 2];

				double t{};
				double u{};
				double v{};

				if (!mesh.intersect(
					origin,
					dir,
					mesh.m_vertices[index0].pos,
					mesh.m_vertices[index1].pos,
					mesh.m_vertices[index2].pos,
					t,
					u,
					v))
				{
					continue;
				}

				closestT = t;
				closestU = u;
				closestV = v;

				closestIndex[0] = index0;
				closestIndex[1] = index1;
				closestIndex[2] = index2;

				closestObj = meshRef.first;

				surfaceIndex = i;
			}
		}

		if (std::holds_alternative<std::monostate>(closestObj)){
			return background;
		}

		return trace(
			closestObj,
			closestIndex,
			surfaceIndex,
			dir,
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
	using SceneObjPtr = std::variant<std::monostate, TriangularMesh*, Light*>;

	template<typename... Ts> struct Overload : Ts...{
		using Ts::operator()...; 
	};

	void transform_all(const Eigen::Matrix4d& cam_transform) {
		for (auto& obj : m_mesh) {
			obj->applyTransform(cam_transform);
		}
		for (auto& obj : m_light){
			obj->applyTransform(cam_transform);
		}
	}

	double trace(
		const SceneObjPtr& obj_ptr,
		const std::size_t* vertice_index,
		const std::size_t& surface_index,
		const Eigen::Vector3d& dir,
		const double& u,
		const double& v,
		const double& t)
	{
		const Light* light_ptr{ nullptr };
		const TriangularMesh* mesh{ nullptr };
		
		{
			auto f1 = [](const std::monostate&) -> const TriangularMesh* {
				return nullptr;
				};
			auto f2 = [&](const Light* const& light) -> const TriangularMesh* {
				light_ptr = light;
				return const_cast<const TriangularMesh*>(light->getMesh().value());
			};
			auto f3 = [](const TriangularMesh* const& mesh) -> const TriangularMesh* {
				return mesh;
				};
			mesh = std::visit(
				Overload<decltype(f1), decltype(f2), decltype(f3)>{f1, f2, f3},
				obj_ptr);
		}
		// This propably can be deleted
		if (mesh == nullptr) {
			std::cout << "This path should never be reached \n";
			return m_background;
		}
		
		const std::complex<double> refractive{
			TriangularMesh::get_Barycentric_Interpolated_refractive_index(
				mesh->m_vertices[vertice_index[0]],
				mesh->m_vertices[vertice_index[1]],
				mesh->m_vertices[vertice_index[2]],
				u,
				v)
		};

		// it might be necessary to inverto one vector 
		double cos_theta{ mesh->m_surface_normals[surface_index].dot(dir) };

		// Only validity check!! should be later removed 
		if (cos_theta < 0.0) {
			std::cout << "cos Theta is negative \n";
			throw std::runtime_error("Check");
		}

		// Hardcoded Display Image !!! 
		if (mesh->m_info.emitter) {
			const Eigen::Vector2d texture_coord{ TriangularMesh::get_Texture_Coord(
			mesh->m_vertices[vertice_index[0]].uv,
			mesh->m_vertices[vertice_index[1]].uv,
			mesh->m_vertices[vertice_index[2]].uv,
			u,
			v) 
			};
			// BRDF characeteristic of Source
			const double scalingBRDF{ BRDF(1.0, refractive).get_Reflection(cos_theta,t) };
			// Local Brightness in [0...1] 
			const double light_Brightness{ light_ptr->get_local_Texture(texture_coord, 0) };
			// Return maxLightPower * Angle- & distance- Scaling * Brightness IF the light is modular 
			return light_ptr->m_info.m_power * scalingBRDF * light_Brightness;
		}

		Eigen::Vector3d reflected = reflect_ray(
			dir,
			(mesh->m_vertices[vertice_index[1]].pos - mesh->m_vertices[vertice_index[0]].pos).cross(
				mesh->m_vertices[vertice_index[2]].pos - mesh->m_vertices[vertice_index[0]].pos));

		const Eigen::Vector3d origin{ t * dir };
		
		
		if (mesh->m_info.specular) {
			const double scaling_BRDF{ BRDF(1.0, refractive).get_Reflection(cos_theta, t) };
			return scaling_BRDF * castRay(
					origin,
					reflected,
					m_mesh,
					m_light,
					m_background
				);
		}

		// Here the same 
		if (mesh->m_info.diffuse) {
			throw std::invalid_argument("We do not have a pipeline for diffuse objects \n");
		}
		return m_background;
	}
	//bool optimizer();

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