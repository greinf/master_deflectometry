#ifndef SCENE_HPP
#define SCENE_HPP

#include "Mesh.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include <vector>
#include <memory>

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
	
	void raytraceScene(std::vector<Eigen::MatrixXd>& out_img) {
		if (m_cam.empty()) {
			throw std::runtime_error("No Camera in the scene");
			return;
		}
		if (m_mesh.empty()) std::cout << "No object in the scene \n";
		if (m_light.empty()) std::cout << "No Light sources in the scene \n";

		m_results.reserve(m_cam.size());
		
		// For all available Objects calculate the surface normals 
		/*for (auto& obj : m_mesh) {
			if (obj->m_surface_normals == nullptr) obj->calc_surface_normals();
		}*/

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
	static double castRay(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const std::vector<std::unique_ptr<TriangularMesh>>& meshes,
		const std::vector<std::unique_ptr<Light>>& lights,
		const double& background)
	{
		double closestT = std::numeric_limits<double>::infinity();

		double closestU{};
		double closestV{};

		std::size_t closestIndex0{};
		std::size_t closestIndex1{};
		std::size_t closestIndex2{};

		const TriangularMesh* closestMesh = nullptr;

		for (const auto& meshPtr : meshes) {
			const TriangularMesh& mesh = *meshPtr;

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

				if (t >= closestT) {
					continue;
				}

				closestT = t;
				closestU = u;
				closestV = v;

				closestIndex0 = index0;
				closestIndex1 = index1;
				closestIndex2 = index2;

				closestMesh = &mesh;
			}
		}

		if (closestMesh == nullptr) {
			return background;
		}

		return trace(
			closestMesh->m_info,
			closestMesh->m_vertices[closestIndex0],
			closestMesh->m_vertices[closestIndex1],
			closestMesh->m_vertices[closestIndex2],
			dir,
			closestT,
			closestU,
			closestV);
	}

	double m_background{};

private:
	std::vector<std::unique_ptr<Camera>> m_cam{};
	std::vector<std::unique_ptr<TriangularMesh>> m_mesh{};
	std::vector<std::unique_ptr<Light>> m_light{};

	std::vector<std::unique_ptr<Eigen::MatrixXd>> m_results;

	void transform_all(const Eigen::Matrix4d& cam_transform) {
		for (const auto& obj : m_mesh) {
			if (!obj->m_transform.isIdentity() ||
				!cam_transform.isIdentity()) {
				obj->applyTransform(cam_transform);
			}
		}
		for (const auto& obj : m_light) {
			if (!obj->m_transform.isIdentity() || 
				!cam_transform.isIdentity()) {
				obj->applyTransform(cam_transform);
			}
		}
	}

	static double trace(
		const ObjectInfo& info,
		const Vertice& v0,
		const Vertice& v1,
		const Vertice& v2,
		const Eigen::Vector3d dir,
		const double& u,
		const double& v,
		const double& t)
	{
		Eigen::Vector3d reflected = reflect_ray(
			dir,
			(v1.pos - v0.pos).cross(v2.pos - v0.pos));

		if (!info.closed) {
			
		}

		

		const double w{ 1 - u - v };
		const double refractive_index{
			w * v0.refractive_index +
			u * v1.refractive_index +
			v * v2.refractive_index
		};


		return true;
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