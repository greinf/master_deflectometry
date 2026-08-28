#ifndef TRACEABLEMESH_HPP	
#define TRACEABLEMESH_HPP

#include <Eigen/dense>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include "Mesh.hpp"
#include "Light.hpp"
#include "ThreadPool.hpp"



// Traceable Mesh builds a wrapper around the 
class TraceAbleMesh {
protected:
	struct Vertex {
		Eigen::Vector3d* pos{ nullptr };
		Eigen::Vector3d* vertex_normal{ nullptr };
		Eigen::Vector2d* uv_Vertice{ nullptr };
		std::complex<double>* refractive{ nullptr };
	};

public:
	double* m_area{};
	struct Triangle {
		Eigen::Vector3d* surf_norm{ nullptr };
		Vertex vertex[3]{};
		double* area{};
	};

	virtual ~TraceAbleMesh() = default;

	std::unique_ptr<Triangle[]> m_triangle{ nullptr };

	TraceAbleMesh(const TraceAbleMesh&) = delete;
	TraceAbleMesh& operator=(const TraceAbleMesh&) = delete;
	TraceAbleMesh(TraceAbleMesh&&) noexcept = delete;
	TraceAbleMesh& operator=(TraceAbleMesh&&) noexcept = delete;

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

	TraceAbleMesh(
		const std::size_t& n_Triangles,
		const Triangle* triangle
	) 
	{
		m_triangle = std::make_unique<Triangle[]>(n_Triangles);
		for (std::size_t i = 0; i < n_Triangles; ++i) {
			m_triangle[i] = triangle[i];
		}
	}

	std::size_t m_n_triangles{};

	virtual bool intersect(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		Triangle*& triangle_out,
		double& t, double& u, double& v
	) const
	{
		double closestT{ std::numeric_limits<double>::infinity() };
		std::size_t counter{};
		IntersectProtocoll closestHit_Protocoll{};

		for (std::size_t i = 0; i < m_n_triangles; ++i) {
			double t_temp{ 0.0 }, u_temp{ 0.0 }, v_temp{ 0.0 };
			IntersectProtocoll protocoll = 
				TraceAbleMesh::intersect_tri(
					origin,
					dir,
					i,
					t_temp,
					u_temp,
					v_temp
			);

			if (protocoll.hit == false) continue;
			if (protocoll.t > closestT)
				continue;

			closestT = protocoll.t;
			closestHit_Protocoll = protocoll;
		}

		if (std::isinf(closestT)) return false;

		// If not Valid pass. Return true and assign the Values to the Function Arguments
		u = closestHit_Protocoll.u;
		v = closestHit_Protocoll.v;
		t = closestHit_Protocoll.t;

		triangle_out = &m_triangle.get()[closestHit_Protocoll.TriangleIndex];
		return true;
	}

	ObjectInfo* m_info{ nullptr };

	std::optional<const Light*> getLight() const {
		if (m_light == nullptr) return std::nullopt;
		return std::optional<const Light*>(m_light);
	}

	struct IntersectProtocoll {
		bool hit{};
		double t{}, u{}, v{};
		std::size_t TriangleIndex{std::numeric_limits<std::size_t>::max()};
	};

	IntersectProtocoll intersect_tri(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const std::size_t triangle_index,
		double& t, double& u, double& v) const
	{
		bool hit = TriangularMesh::intersect(
			origin, 
			dir, 
			*m_triangle[triangle_index].vertex[0].pos, 
			*m_triangle[triangle_index].vertex[1].pos,
			*m_triangle[triangle_index].vertex[2].pos,
			t, 
			u, 
			v);

		IntersectProtocoll protocoll{
			{hit},
			{t}, {u}, {v},
			{triangle_index}
		};

		return protocoll;
	}

protected:
	TriangularMesh* m_mesh{ nullptr };
	Light* m_light{ nullptr };

private:
	std::once_flag flag;
	
	void build(const std::size_t& n_triangles, const Triangle* triangle) {
		m_n_triangles = n_triangles;

		m_triangle = std::unique_ptr<Triangle[]>(new Triangle[m_n_triangles]);

		for (std::size_t i = 0; i < n_triangles; ++i)
		{
			m_triangle[i] = *triangle;
		}
	}

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
			m_area = &m_mesh->m_completeArea;
		}
	}
};


#endif // TRACEABLEMESH_HPP
