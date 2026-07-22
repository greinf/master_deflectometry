//#ifndef BVH_HPP
//#define BVH_HPP
//
//#include "Light.hpp"
//#include "Mesh.hpp"
//#include <vector>
//#include <limits>
//
//using SceneObjPtr = std::variant<std::monostate, TriangularMesh*, Light*>;
//
//class BVH : public TriangularMesh, public Light {
//private:
//	struct AABB {
//		Eigen::Vector3d min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
//		Eigen::Vector3d max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
//	};
//
//	struct BVH_Node {
//		std::vector<BVH_Node*> child_nodes{};
//		bool m_isRoot{ false };
//		AABB m_Bounding{};
//	};
//
//	class BVH_Tree {
//		std::size_t m_n_leafs{};
//		BVH_Node* m_rootNode{ nullptr };
//
//		void insert() {};
//	};
//
//	void extract() {
//		if (std::holds_alternative<TriangularMesh>(m_Object)) {
//			m_mesh = std::get<TriangularMesh*>(m_Object)
//		}
//	}
//	
//
//	void build() {
//		extract()
//	}
//
//	SceneObjPtr m_Object;
//	TriangularMesh* m_mesh{ nullptr };
//
//
//public:
//	BVH(const SceneObjPtr& obj)
//		:m_Object{ obj }
//	{
//		if (std::holds_alternative<std::monostate>(obj)) 
//			throw std::invalid_argument("Empty Varinat for BVH");
//		build();
//	}
//};
//
//#endif // "BVH_HPP"