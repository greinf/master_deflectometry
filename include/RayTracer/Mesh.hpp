#ifndef MESH_HPP
#define MESH_HPP
#include <Eigen/dense>
#include "Object.hpp"
#include <vector>
#include <iterator>
#include <algorithm>
#include <open3d/geometry/TriangleMesh.h>
#include <cstdlib>

struct ObjectInfo {
	bool closed{ false };
	bool specular{ true };
	double area{ 0 };
};

// Utilities 
struct Vertice {
	Eigen::Vector3d pos{};
	double refractive_index{};
	// if not specifically marked objects are specular
};

class PolygonMesh;
class TriangularMesh;

class Mesh : public Object {
public:
	Mesh(const Mesh&) = delete;
	Mesh& operator=(const Mesh&) = delete;
	Mesh() = default;

	Mesh& operator=(Mesh&& mesh) noexcept {
		std::swap(m_vertices, mesh.m_vertices);
		std::swap(m_vertice_normals, mesh.m_vertice_normals);
		std::swap(m_indices, mesh.m_indices);
		std::swap(m_n_ver_per_surface, mesh.m_n_ver_per_surface);
		std::swap(m_n_vertices, mesh.m_n_vertices);
		std::swap(m_n_surfaces, mesh.m_n_surfaces);
		return *this;
	}

	explicit Mesh(std::unique_ptr<Vertice[]>&& vertices,
		std::unique_ptr<Eigen::Vector3d[]>&& vertice_normal,
		std::unique_ptr<std::uint32_t[]>&& indices,
		std::unique_ptr<std::uint32_t[]>&& n_ver_per_surface,
		const std::uint32_t& n_vertice,
		const std::uint32_t& n_surface)
		: Object()
		, m_vertices{ std::move(vertices) }
		, m_vertice_normals{ std::move(vertice_normal) }
		, m_indices{ std::move(indices) }
		, m_n_ver_per_surface{ std::move(n_ver_per_surface) }
		, m_n_vertices{n_vertice}
		, m_n_surfaces{n_surface}
	{ }

	Mesh(Mesh&& mesh) noexcept
		: Object()
		, m_vertices{ std::move(mesh.m_vertices) }
		, m_vertice_normals{ std::move(mesh.m_vertice_normals) }
		, m_indices{ std::move(mesh.m_indices) }
		, m_n_ver_per_surface{ std::move(mesh.m_n_ver_per_surface) }
		, m_n_vertices{ std::move(mesh.m_n_vertices) }
		, m_n_surfaces{ std::move(mesh.m_n_surfaces) }
	{ }

	Mesh(std::unique_ptr<Mesh>&& mesh)
		: Object()
		, m_vertices{std::move(mesh->m_vertices)}
		, m_vertice_normals{std::move(mesh->m_vertice_normals)}
		, m_indices{std::move(mesh->m_indices)}
		, m_n_ver_per_surface{std::move(mesh->m_n_ver_per_surface)}
		, m_n_vertices{std::move(mesh->m_n_vertices)}
		, m_n_surfaces{std::move(mesh->m_n_surfaces)}
	{ }

	virtual double get_Area() const noexcept = 0;

	void applyTransform(const Eigen::Matrix4d& cam)
	{
		Eigen::Matrix4d transform{ cam * m_transform };
		Eigen::Matrix3d normal_transform{ transform.block<3,3>(0,0).transpose().inverse() };
		for (std::size_t i = 0; i < static_cast<std::size_t>(m_n_vertices); ++i)
		{
			Eigen::Vector4d homogenous{};
			homogenous.head<3>() = m_vertices[i].pos;
			homogenous(3) = 1.0;
			m_vertices[i].pos = (transform * homogenous).head<3>();

			m_vertice_normals[i] = normal_transform * m_vertice_normals[i];
		}
		m_transform = Eigen::Matrix4d::Identity();
	} virtual

	void calc_surface_normals()
	{
		if (!m_n_surfaces) {
			std::cout << "Mesh contains zero surfaces \n";
			return;
		}

		std::unique_ptr<Eigen::Vector3d[]>
			surface_normals_temp{ new Eigen::Vector3d[m_n_surfaces] };

		Vertice* vert_ptr{ m_vertices.get() };

		for (std::size_t i = 0; i < static_cast<std::size_t>(m_n_surfaces); ++i) {
			Eigen::Vector3d ab = vert_ptr[i + 1].pos - vert_ptr[i].pos;
			Eigen::Vector3d ac = vert_ptr[i + 2].pos - vert_ptr[i].pos;
			surface_normals_temp.get()[i] = ab.cross(ac);
			vert_ptr += m_n_ver_per_surface[i];
		}
		std::swap(m_surface_normals, surface_normals_temp);
	}

	virtual bool shade() const = 0;

	virtual bool intersect(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const Eigen::Vector3d& v0, 
		const Eigen::Vector3d& v1,
		const Eigen::Vector3d& v2,
		double& t, double &u, double &v) const = 0;

	~Mesh() override = default;

	std::unique_ptr<Vertice[]> m_vertices = nullptr;
	std::unique_ptr<Eigen::Vector3d[]> m_vertice_normals = nullptr;
	std::unique_ptr<Eigen::Vector3d[]> m_surface_normals = nullptr;
	std::unique_ptr<std::uint32_t[]> m_indices = nullptr;
	std::unique_ptr<std::uint32_t[]> m_n_ver_per_surface = nullptr;
	std::uint32_t m_n_vertices{};
	std::uint32_t m_n_surfaces{};
	ObjectInfo m_info{};
};

class TriangularMesh : public Mesh {
public:
	explicit TriangularMesh(
		std::unique_ptr<Vertice[]>&& vertices,
		std::unique_ptr<Eigen::Vector3d[]>&& vertice_normals,
		std::unique_ptr<std::uint32_t[]>&& indices,
		std::unique_ptr<std::uint32_t[]>&& n_ver_per_surface,
		const std::uint32_t& n_vertices,
		const std::uint32_t& n_surfaces)
		:Mesh(
			std::move(vertices),
			std::move(vertice_normals),
			std::move(indices),
			std::move(n_ver_per_surface),
			n_vertices,
			n_surfaces)
	{ }

	TriangularMesh()
		:Mesh()
	{ }

	double get_Area() const noexcept override 
	{
		double area{};
		const Vertice* vert_ptr{ m_vertices.get() };
		for (std::size_t i = 0; i < m_n_surfaces; ++i) {
			const Eigen::Vector3d b_a{ vert_ptr[i + 1].pos - vert_ptr[i].pos };
			const Eigen::Vector3d c_a{ vert_ptr[i + 2].pos - vert_ptr[i].pos };
			area += 1 / 2 * b_a.cross(c_a).norm();
			vert_ptr += m_n_ver_per_surface[i];
		}
		return area;
	}


	bool shade() const override
	{
		throw std::runtime_error("not implemented");
		return false;
	}

	bool intersect(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const Eigen::Vector3d& v0, // A
		const Eigen::Vector3d& v1, // B
		const Eigen::Vector3d& v2, // C
		double& t, double& u, double& v) const override
	{
		// Truber-Moller algorithm
		const Eigen::Vector3d AB{ v1 - v0 };
		const Eigen::Vector3d AC{ v2 - v0 };
		const Eigen::Vector3d R{ origin - v0 };
		
		constexpr double eps{ 1e-9 };

		// Matrix mat(e1, e2, e3): det(mat) -> (e1 x e2) * e3
		// Important clarification:
		// a * (b x c) = (a x b) * c; 
		// a * (b x c) = b * (c x a) = c * (a x b) 

		// P = O + t*D
		// P = A + u(B-A) + v (C-A)
		// O + t*D = A + u(B-A) + v(C-A)
		// O - A = -tD + u(B-A) + v(C-A)
		// [-D (B-A) (C-A)] * [t u v]^T = O - A
		
		const Eigen::Vector3d e1xe2 = -dir.cross(AB);
		const Eigen::Vector3d e2xe3 = AB.cross(AC);
		const Eigen::Vector3d e3xe1 = AC.cross(-dir);

		// For cramers Rule we create the determinantes for each e1 e2 e3 -> det_t det_u det_v
		const double det{ e1xe2.dot(AC) };
		if (det < eps) return false;

		const double inv_det{ 1.0 / det };

		// u - value
		const double det_u{ e3xe1.dot(R) };
		u = det_u * inv_det;
		if (u < 0.0 || u > 1.0) return false;

		// t - value
		const double det_t{ e2xe3.dot(R) };
		t = det_t * inv_det;
		if (t < -eps) return false;

		// v- value
		const double det_v{ e1xe2.dot(R) };
		v = det_v * inv_det;
		if (v < 0.0 || v > 1.0) return false;
		
		if (u + v > 1.0) return false;
		
		return true;
	}

	operator open3d::geometry::TriangleMesh() const
	{
		if (m_n_vertices == 0) {
			throw std::runtime_error("Mesh has no vertices.");
		}

		if (m_n_surfaces == 0) {
			throw std::runtime_error("Mesh has no surfaces.");
		}

		if (!m_vertices) {
			throw std::runtime_error("Vertex array is null.");
		}

		if (!m_indices) {
			throw std::runtime_error("Index array is null.");
		}

		std::vector<Eigen::Vector3d> vertices(m_n_vertices);
		std::vector<Eigen::Vector3i> triangles(m_n_surfaces);

		for (std::size_t i = 0; i < m_n_vertices; ++i) {
			vertices[i] = m_vertices[i].pos;
		}

		for (std::size_t i = 0; i < m_n_surfaces; ++i) {
			const auto i0 = m_indices[i * 3];
			const auto i1 = m_indices[i * 3 + 1];
			const auto i2 = m_indices[i * 3 + 2];

			if (i0 >= m_n_vertices ||
				i1 >= m_n_vertices ||
				i2 >= m_n_vertices) {
				throw std::runtime_error(
					"Triangle contains an invalid vertex index."
				);
			}

			triangles[i] = Eigen::Vector3i{
				static_cast<int>(i0),
				static_cast<int>(i1),
				static_cast<int>(i2)
			};
		}

		open3d::geometry::TriangleMesh result{
			std::move(vertices),
			std::move(triangles)
		};

		if (m_vertice_normals) {
			result.vertex_normals_.resize(m_n_vertices);

			for (std::size_t i = 0; i < m_n_vertices; ++i) {
				result.vertex_normals_[i] = m_vertice_normals[i];
			}
		}

		return result;
	}

private:
	
};

class PolygonMesh : public Mesh {
public:
	
	PolygonMesh(const PolygonMesh&) = delete;
	PolygonMesh& operator=(const PolygonMesh&) = delete;

	PolygonMesh()
		:Mesh()
	{ }

	explicit PolygonMesh(
		std::unique_ptr<Vertice[]>&& vertices,
		std::unique_ptr<Eigen::Vector3d[]>&& vertice_normal,
		std::unique_ptr<std::uint32_t[]>&& indices,
		std::unique_ptr<std::uint32_t[]>&& n_ver_per_surface,
		const std::uint32_t n_vertices,
		const std::uint32_t n_surfaces)
		: Mesh(
			std::move(vertices), 
			std::move(vertice_normal),
			std::move(indices),
			std::move(n_ver_per_surface),
			n_vertices,
			n_surfaces)
	{ }

	bool shade() const override {
		throw std::runtime_error("Not implemented");
		return false;
	}

	double get_Area() const noexcept override {
		const auto tri_mesh{ this->convert2Triangular() };
		return tri_mesh->get_Area();
	}

	bool intersect(
		const Eigen::Vector3d& origin,
		const Eigen::Vector3d& dir,
		const Eigen::Vector3d& v0,
		const Eigen::Vector3d& v1,
		const Eigen::Vector3d& v2,
		double& t, double& u, double& v) const override
	{
		throw std::runtime_error("Not implemented");
		return false;
	}

	std::unique_ptr<TriangularMesh> convert2Triangular() const {
		const std::size_t n_surfaces{ static_cast<std::size_t>(m_n_surfaces) };

		// Build Tempory Container 
		std::vector<std::uint32_t> n_ver_per_surface_temp;
		n_ver_per_surface_temp.reserve(n_surfaces * 2);
		std::vector<std::uint32_t> indizes_temp;
		indizes_temp.reserve(n_surfaces * 3);

		// Helper
		auto extend = [](
			std::vector<std::uint32_t>& vertice_surf,
			std::uint32_t n_vertice_surf,
			std::vector<std::uint32_t>& indizes,
			const std::uint32_t* tri_indiz)
			{
				vertice_surf.push_back(n_vertice_surf);

				for (std::size_t i = 0; i < n_vertice_surf; ++i)
				{
					indizes.push_back(tri_indiz[i]);
				}
			};

		std::uint32_t* index_ptr{ m_indices.get() };

		// Fill the temporary container  
		for (std::size_t i = 0; i < n_surfaces; ++i)
		{
			if (m_n_ver_per_surface[i] == 3) {
				std::uint32_t indizes[3]{};
				for (std::size_t i = 0; i < 3; ++i) {
					indizes[i] = *index_ptr;
					++index_ptr;
				}

				extend(
					n_ver_per_surface_temp,
					3,
					indizes_temp,
					indizes
				);
			}
			
			else if (m_n_ver_per_surface[i] > 3) {
				const std::size_t n_vertices{ static_cast<std::size_t>(m_n_ver_per_surface[i]) };
				std::vector<std::uint32_t> indizes_poly(n_vertices);
				std::uint32_t indizes[3]{};

				// Create container of available vectors
				for (std::size_t u = 0; u < n_vertices; ++u) {
					indizes_poly[u] = *index_ptr;
					++index_ptr;
				}
				
				// Bring the indizes in the new order
				for (std::size_t j = 0; j + 2 < n_vertices; ++j) {
					indizes[0] = indizes_poly[0];
					indizes[1] = indizes_poly[j + 1];
					indizes[2] = indizes_poly[j + 2];

					extend(
						n_ver_per_surface_temp,
						3,
						indizes_temp,
						indizes
					);
				}
			}
		}
		
		// Refill the Containers
		const std::size_t ver_per_surf_size{ n_ver_per_surface_temp.size() };
		const std::size_t indizes_size{ indizes_temp.size() };

		std::unique_ptr<std::uint32_t[]> temp_ver_per_surf_swap{ new std::uint32_t[ver_per_surf_size] };
		std::unique_ptr<std::uint32_t[]> temp_indizes_swap{ new std::uint32_t[indizes_size] };

		{
			std::uint32_t* temp_ver_per_surf_ptr{ temp_ver_per_surf_swap.get() };
			for (const auto& element : n_ver_per_surface_temp) {
				*temp_ver_per_surf_ptr = element;
				++temp_ver_per_surf_ptr;
			}

			std::uint32_t* temp_indizes_ptr{ temp_indizes_swap.get() };
			for (const auto& element : indizes_temp) {
				*temp_indizes_ptr = element;
				++temp_indizes_ptr;
			}
		}

		// Create deepcopy
		std::unique_ptr<Vertice[]> 
			vertice_copy{ new Vertice[static_cast<std::size_t>(m_n_vertices)] };
		std::unique_ptr<Eigen::Vector3d[]> 
			vertice_normal_copy{ new Eigen::Vector3d[static_cast<std::size_t>(m_n_vertices)] };

		{
			Vertice* vertice_copy_ptr{ vertice_copy.get() };
			Vertice* vertices_ptr{ m_vertices.get() };
			Eigen::Vector3d* vertice_normal_copy_ptr{ vertice_normal_copy.get() };
			Eigen::Vector3d* vertice_normal_ptr{ m_vertice_normals.get() };

			for (std::size_t i = 0; i < static_cast<std::size_t>(m_n_vertices); ++i) {
				vertice_copy_ptr[i] = vertices_ptr[i];
				vertice_normal_copy_ptr[i] = vertice_normal_ptr[i];
			}
		}

		const auto n_triangles =
			static_cast<std::uint32_t>(n_ver_per_surface_temp.size());

		// It might be interesting to share the vertice points between different representations (polygonMesh and Triangular)
		
		return std::make_unique<TriangularMesh>(
			std::move(vertice_copy),
			std::move(vertice_normal_copy),
			std::move(temp_indizes_swap),
			std::move(temp_ver_per_surf_swap),
			m_n_vertices,
			n_triangles
			);
	}

	// CoordianteSystem placed in upper left corner! 
	static std::unique_ptr<PolygonMesh> Display(
		const int& pixel_x = 1920,
		const int& pixel_y = 1080,
		const double& pixel_pitch = 0.2745)
	{
		if (pixel_x < 0 || pixel_y < 0 || pixel_pitch < 0) 
			throw std::invalid_argument("Display Parameters must be bigger than zero \n");

		const std::size_t n_surfaces{ 1 };
		const std::size_t n_vertices{ 4 };
		const std::size_t n_indices{ 4 };

		std::unique_ptr<Vertice[]> vertice_temp{ new Vertice[n_vertices] };
		std::unique_ptr<Eigen::Vector3d[]> vertex_normals_temp{ new Eigen::Vector3d[n_vertices] };
		std::unique_ptr<std::uint32_t[]> n_ver_per_surf_temp{ new std::uint32_t[n_surfaces] };
		std::unique_ptr<std::uint32_t[]> indizes_temp{ new std::uint32_t[n_vertices] };

		// Create Vertices
		vertice_temp[0].pos = Eigen::Vector3d::Zero();
		vertice_temp[1].pos = Eigen::Vector3d(0.0, pixel_y * pixel_pitch, 0.0);
		vertice_temp[2].pos = Eigen::Vector3d(pixel_x * pixel_pitch, pixel_y * pixel_pitch, 0.0);
		vertice_temp[3].pos = Eigen::Vector3d(pixel_x * pixel_pitch, 0.0, 0.0);

		// Connectivity
		vertex_normals_temp[0] = Eigen::Vector3d(0.0, 0.0, 1.0);
		vertex_normals_temp[1] = vertex_normals_temp[0];
		vertex_normals_temp[2] = vertex_normals_temp[1];
		vertex_normals_temp[3] = vertex_normals_temp[2];

		n_ver_per_surf_temp[0] = 4;

		indizes_temp[0] = 0;
		indizes_temp[1] = 1;
		indizes_temp[2] = 2;
		indizes_temp[3] = 3;

		return std::make_unique<PolygonMesh>(
			std::move(vertice_temp),
			std::move(vertex_normals_temp),
			std::move(indizes_temp),
			std::move(n_ver_per_surf_temp),
			static_cast<std::uint32_t>(n_vertices),
			static_cast<std::uint32_t>(n_surfaces)
		);

		//const std::size_t n_surfaces{ static_cast<std::size_t>(pixel_x * pixel_y) };
		//const std::size_t n_vertices{ static_cast<std::size_t>(2 + 2 * n_surfaces * n_surfaces) };
		//const std::size_t n_indices{ static_cast<std::size_t>(n_surfaces * 4) };

		//std::unique_ptr<Vertice[]> vertice_temp{ new Vertice[n_vertices] };
		//std::unique_ptr<Eigen::Vector3d[]> vertex_normals_temp{new Eigen::Vector3d[n_vertices]};
		//std::unique_ptr<std::uint32_t[]> n_ver_per_surf_temp{ new std::uint32_t[n_surfaces] };
		//std::unique_ptr<std::uint32_t[]> indices_temp{ new std::uint32_t[n_indices] };

		//// Create Vertices 
		//for (std::size_t i = 0; i < n_vertices; ++i) {
		//	std::div_t coords{ std::div(static_cast<int>(i), pixel_x) };
		//	vertice_temp[i].pos =
		//		Eigen::Vector3d(
		//			coords.rem * pixel_pitch,
		//			coords.quot * pixel_pitch,
		//			0.0);
		//	vertex_normals_temp[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
		//}

		//// Connectivity 
		//for (std::size_t i = 0; i < n_surfaces; ++i) {
		//	n_ver_per_surf_temp[i] = 4;
		//	
		//	std::div_t coord_helper{ std::div(static_cast<int>(i), pixel_x) };
		//	indices_temp[i * 4] = static_cast<std::uint32_t>
		//		(coord_helper.rem + coord_helper.quot * pixel_x);
		//	indices_temp[i * 4 + 1] = static_cast<std::uint32_t>
		//		(coord_helper.rem + (coord_helper.quot + 1) * pixel_x);
		//	indices_temp[i * 4 + 2] = static_cast<std::uint32_t>
		//		((coord_helper.rem+1) + (coord_helper.quot + 1) * pixel_x);
		//	indices_temp[i * 4 + 3] = static_cast<std::uint32_t>
		//		((coord_helper.rem + 1) + (coord_helper.quot) * pixel_x);
		//}
	}
	
	static std::unique_ptr<PolygonMesh> ParabolicalMirror(
		double focal_length,
		double max_r,
		int division)
	{
		if (division < 2) {
			throw std::invalid_argument(
				"division must be at least 2."
			);
		}

		if (focal_length == 0.0) {
			throw std::invalid_argument(
				"focal_length must not be zero."
			);
		}

		const std::size_t div =
			static_cast<std::size_t>(division);

		const std::size_t n_vertices =
			1 + (div - 1) * div;

		const std::size_t n_surfaces =
			div + div * (div - 2);

		const std::size_t n_indices =
			div * 3 + div * (div - 2) * 4;

		auto vertices =
			std::make_unique<Vertice[]>(n_vertices);

		auto vertex_normals =
			std::make_unique<Eigen::Vector3d[]>(n_vertices);

		auto indices =
			std::make_unique<std::uint32_t[]>(n_indices);

		auto vertices_per_surface =
			std::make_unique<std::uint32_t[]>(n_surfaces);

		constexpr double two_pi = 2.0 * M_PI;
		const double r_step = max_r / static_cast<double>(div);

		// Center vertex
		vertices[0].pos = Eigen::Vector3d{ 0.0, 0.0, 0.0 };

		// Circular rings
		std::size_t vertex_index = 1;

		for (std::size_t r = 1; r < div; ++r) {
			const double radius =
				r_step * static_cast<double>(r);

			for (std::size_t theta = 0; theta < div; ++theta) {
				const double angle =
					two_pi * static_cast<double>(theta)
					/ static_cast<double>(div);

				vertices[vertex_index].pos = Eigen::Vector3d{
					radius * std::cos(angle),
					radius * std::sin(angle),
					radius * radius / (4.0 * focal_length)
				};

				++vertex_index;
			}
		}

		std::size_t surface_index = 0;
		std::size_t index_offset = 0;

		// Center triangles
		for (std::size_t theta = 0; theta < div; ++theta) {
			const std::size_t next_theta = (theta + 1) % div;

			vertices_per_surface[surface_index++] = 3;

			indices[index_offset++] = 0;
			indices[index_offset++] =
				static_cast<std::uint32_t>(1 + theta);
			indices[index_offset++] =
				static_cast<std::uint32_t>(1 + next_theta);
		}

		// Quads between rings
		for (std::size_t r = 2; r < div; ++r) {
			const std::uint32_t previous_start =
				static_cast<std::uint32_t>(
					1 + (r - 2) * div
					);

			const std::uint32_t current_start =
				static_cast<std::uint32_t>(
					1 + (r - 1) * div
					);

			for (std::size_t theta = 0; theta < div; ++theta) {
				const std::size_t next_theta = (theta + 1) % div;

				vertices_per_surface[surface_index++] = 4;

				indices[index_offset++] =
					previous_start
					+ static_cast<std::uint32_t>(theta);

				indices[index_offset++] =
					current_start
					+ static_cast<std::uint32_t>(theta);

				indices[index_offset++] =
					current_start
					+ static_cast<std::uint32_t>(next_theta);

				indices[index_offset++] =
					previous_start
					+ static_cast<std::uint32_t>(next_theta);
			}
		}

		// Vertex normals
		for (std::size_t i = 0; i < n_vertices; ++i) {
			const double x = vertices[i].pos.x();
			const double y = vertices[i].pos.y();

			vertex_normals[i] = Eigen::Vector3d{
				-x / (2.0 * focal_length),
				-y / (2.0 * focal_length),
				1.0
			}.normalized();
		}

		return std::make_unique<PolygonMesh>(
			std::move(vertices),
			std::move(vertex_normals),
			std::move(indices),
			std::move(vertices_per_surface),
			static_cast<std::uint32_t>(n_vertices),
			static_cast<std::uint32_t>(n_surfaces)
		);
	}

	~PolygonMesh() override = default;

private:
	std::unique_ptr<TriangularMesh> m_triangle{ nullptr };
};

#endif //"Mesh"