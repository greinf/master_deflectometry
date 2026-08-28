#ifndef BVH_HPP
#define BVH_HPP

#include "TraceAlbeMesh.hpp"
#include "Mesh.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>


class BVH : public TraceAbleMesh
{
private:
    static constexpr std::size_t kDirectionCount{ 7 };
    static constexpr std::size_t kLeafSize{ 20 };
    static constexpr std::size_t kTraversalStackSize{ 64 };

    static constexpr std::uint32_t kInvalidNode{
        std::numeric_limits<std::uint32_t>::max()
    };

    static inline const std::array<Eigen::Vector3d, kDirectionCount>
        s_boundingDirections{ {
            Eigen::Vector3d{ 1.0,  0.0,  0.0 },
            Eigen::Vector3d{ 0.0,  1.0,  0.0 },
            Eigen::Vector3d{ 0.0,  0.0,  1.0 },

            Eigen::Vector3d{ 1.0,  1.0, 1.0 }.normalized(),
            Eigen::Vector3d{-1.0,  1.0, 1.0 }.normalized(),
            Eigen::Vector3d{-1.0, -1.0, 1.0 }.normalized(),
            Eigen::Vector3d{ 1.0, -1.0, 1.0 }.normalized()
        } };


    struct ProjectedRay
    {
        std::array<double, kDirectionCount> origin{};
        std::array<double, kDirectionCount> inverseDirection{};
        std::array<bool, kDirectionCount> parallel{};
    };


    [[nodiscard]] static ProjectedRay projectRay(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction) noexcept
    {
        constexpr double epsilon{ 1e-12 };

        ProjectedRay projectedRay{};

        for (std::size_t i = 0; i < kDirectionCount; ++i)
        {
            projectedRay.origin[i] =
                s_boundingDirections[i].dot(origin);

            const double projectedDirection{
                s_boundingDirections[i].dot(direction)
            };

            projectedRay.parallel[i] =
                std::abs(projectedDirection) < epsilon;

            if (!projectedRay.parallel[i])
            {
                projectedRay.inverseDirection[i] =
                    1.0 / projectedDirection;
            }
        }

        return projectedRay;
    }


    struct BoundingBox
    {
        std::array<double, kDirectionCount> min{};
        std::array<double, kDirectionCount> max{};

        BoundingBox()
        {
            reset();
        }

        void reset() noexcept
        {
            min.fill(std::numeric_limits<double>::infinity());
            max.fill(-std::numeric_limits<double>::infinity());
        }

        [[nodiscard]] bool isValid() const noexcept
        {
            for (std::size_t i = 0; i < kDirectionCount; ++i)
            {
                if (min[i] > max[i])
                {
                    return false;
                }
            }

            return true;
        }

        void expand(const Eigen::Vector3d& point) noexcept
        {
            for (std::size_t i = 0; i < kDirectionCount; ++i)
            {
                const double projection{
                    s_boundingDirections[i].dot(point)
                };

                min[i] = std::min(min[i], projection);
                max[i] = std::max(max[i], projection);
            }
        }

        void expand(const BoundingBox& other) noexcept
        {
            for (std::size_t i = 0; i < kDirectionCount; ++i)
            {
                min[i] = std::min(min[i], other.min[i]);
                max[i] = std::max(max[i], other.max[i]);
            }
        }

        [[nodiscard]] std::size_t longestDirectionIndex() const
        {
            if (!isValid())
            {
                throw std::runtime_error(
                    "Cannot determine longest dimension of invalid bounding box."
                );
            }

            std::size_t longestIndex{ 0 };
            double longestExtent{ max[0] - min[0] };

            for (std::size_t i = 1; i < kDirectionCount; ++i)
            {
                const double extent{ max[i] - min[i] };

                if (extent > longestExtent)
                {
                    longestExtent = extent;
                    longestIndex = i;
                }
            }

            return longestIndex;
        }

        [[nodiscard]] bool intersect(
            const ProjectedRay& ray,
            double maxDistance =
            std::numeric_limits<double>::infinity()) const noexcept
        {
            double tNear{ 0.0 };
            double tFar{ maxDistance };

            for (std::size_t i = 0; i < kDirectionCount; ++i)
            {
                if (ray.parallel[i])
                {
                    if (ray.origin[i] < min[i] ||
                        ray.origin[i] > max[i])
                    {
                        return false;
                    }

                    continue;
                }

                double t0{
                    (min[i] - ray.origin[i]) * ray.inverseDirection[i]
                };

                double t1{
                    (max[i] - ray.origin[i]) * ray.inverseDirection[i]
                };

                if (t0 > t1)
                {
                    std::swap(t0, t1);
                }

                tNear = std::max(tNear, t0);
                tFar = std::min(tFar, t1);

                if (tNear > tFar)
                {
                    return false;
                }
            }

            return tFar >= 0.0;
        }
    };


    struct BVHNode
    {
        BoundingBox bounds{};

        std::uint32_t leftChild{ kInvalidNode };
        std::uint32_t rightChild{ kInvalidNode };

        /*
         * Ein Leaf speichert einen zusammenhängenden Bereich
         * in m_triangleIndices.
         */
        std::size_t firstTriangle{ 0 };
        std::size_t triangleCount{ 0 };

        [[nodiscard]] bool isLeaf() const noexcept
        {
            return leftChild == kInvalidNode &&
                rightChild == kInvalidNode;
        }
    };


    std::vector<BVHNode> m_nodes{};
    //std::vector<std::size_t>
    std::vector<std::size_t> m_triangleIndices{};


public:
    BVH() = delete;

    BVH(const BVH&) = delete;
    BVH(BVH&&) noexcept = delete;

    BVH& operator=(const BVH&) = delete;
    BVH& operator=(BVH&&) noexcept = delete;

    ~BVH() override = default;

    explicit BVH(TriangularMesh* mesh)
        : TraceAbleMesh(mesh)
    {
        buildTree();
    }

    explicit BVH(Light* light)
        : TraceAbleMesh(light)
    {
        buildTree();
    }


    [[nodiscard]] bool intersect(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        Triangle*& triangleOut,
        double& tOut,
        double& uOut,
        double& vOut) const override
    {
        triangleOut = nullptr;

        tOut = std::numeric_limits<double>::infinity();
        uOut = 0.0;
        vOut = 0.0;

        if (m_nodes.empty())
        {
            return false;
        }

        const ProjectedRay projectedRay{
            projectRay(origin, direction)
        };

        std::array<std::uint32_t, kTraversalStackSize> nodeStack{};
        std::size_t stackSize{ 0 };
        nodeStack[stackSize++] = 0;

        bool hitFound{ false };

        while (stackSize > 0)
        {
            const std::uint32_t nodeIndex{
                nodeStack[--stackSize]
            };

            const BVHNode& node{ m_nodes[nodeIndex] };

            /*
             * tOut wird als maximale Entfernung verwendet.
             * Sobald bereits ein Dreieck getroffen wurde, werden
             * weiter entfernte Nodes direkt verworfen.
             */
            if (!node.bounds.intersect(projectedRay, tOut))
            {
                continue;
            }

            if (node.isLeaf())
            {
                const std::size_t end{
                    node.firstTriangle + node.triangleCount
                };

                for (std::size_t i = node.firstTriangle;
                    i < end;
                    ++i)
                {
                    const std::size_t triangleIndex{
                        m_triangleIndices[i]
                    };

                    Triangle& triangle{
                        m_triangle[triangleIndex]
                    };

                    double t{};
                    double u{};
                    double v{};

                    if (!intersectTriangle(
                        origin,
                        direction,
                        triangle,
                        t,
                        u,
                        v))
                    {
                        continue;
                    }

                    if (t >= tOut)
                    {
                        continue;
                    }

                    hitFound = true;

                    tOut = t;
                    uOut = u;
                    vOut = v;

                    triangleOut = &triangle;
                }

                continue;
            }

            if (node.leftChild != kInvalidNode)
            {
                if (stackSize >= nodeStack.size())
                {
                    throw std::runtime_error(
                        "BVH traversal stack capacity exceeded."
                    );
                }

                nodeStack[stackSize++] = node.leftChild;
            }

            if (node.rightChild != kInvalidNode)
            {
                if (stackSize >= nodeStack.size())
                {
                    throw std::runtime_error(
                        "BVH traversal stack capacity exceeded."
                    );
                }

                nodeStack[stackSize++] = node.rightChild;
            }
        }

        return hitFound;
    }


private:
    void buildTree()
    {
        m_nodes.clear();
        m_triangleIndices.clear();

        if (m_n_triangles == 0 || m_triangle == nullptr)
        {
            return;
        }

        m_triangleIndices.resize(m_n_triangles);

        std::iota(
            m_triangleIndices.begin(),
            m_triangleIndices.end(),
            std::size_t{ 0 }
        );

        /*
         * Bei einem binären Baum werden höchstens ungefähr
         * 2N Nodes benötigt.
         */
        m_nodes.reserve(2 * m_n_triangles);

        buildNode(0, m_triangleIndices.size());
    }


    std::uint32_t buildNode(
        const std::size_t begin,
        const std::size_t end)
    {
        if (begin >= end)
        {
            throw std::invalid_argument(
                "Cannot build BVH node from an empty triangle range."
            );
        }

        const std::uint32_t currentNodeIndex{
            static_cast<std::uint32_t>(m_nodes.size())
        };

        m_nodes.emplace_back();

        const std::size_t triangleCount{ end - begin };

        BoundingBox bounds{
            buildBoundingBox(begin, end)
        };

        if (triangleCount <= kLeafSize)
        {
            BVHNode& node{ m_nodes[currentNodeIndex] };

            node.bounds = std::move(bounds);
            node.firstTriangle = begin;
            node.triangleCount = triangleCount;

            return currentNodeIndex;
        }

        const std::size_t directionIndex{
            bounds.longestDirectionIndex()
        };

        const Eigen::Vector3d& splitDirection{
            s_boundingDirections[directionIndex]
        };

        const std::size_t middle{
            begin + triangleCount / 2
        };

        /*
         * Median-Split anhand der projizierten Dreieckszentren.
         */
        std::nth_element(
            m_triangleIndices.begin() +
            static_cast<std::ptrdiff_t>(begin),

            m_triangleIndices.begin() +
            static_cast<std::ptrdiff_t>(middle),

            m_triangleIndices.begin() +
            static_cast<std::ptrdiff_t>(end),

            [this, &splitDirection](
                const std::size_t lhs,
                const std::size_t rhs)
            {
                const double lhsProjection{
                    splitDirection.dot(triangleCenter(lhs))
                };

                const double rhsProjection{
                    splitDirection.dot(triangleCenter(rhs))
                };

                return lhsProjection < rhsProjection;
            }
        );

        const std::uint32_t leftChild{
            buildNode(begin, middle)
        };

        const std::uint32_t rightChild{
            buildNode(middle, end)
        };

        /*
         * Erst nach den rekursiven Aufrufen erneut über den Index
         * auf den Node zugreifen.
         */
        BVHNode& node{ m_nodes[currentNodeIndex] };

        node.bounds = std::move(bounds);
        node.leftChild = leftChild;
        node.rightChild = rightChild;

        return currentNodeIndex;
    }


    [[nodiscard]] BoundingBox buildBoundingBox(
        const std::size_t begin,
        const std::size_t end) const
    {
        BoundingBox bounds{};

        for (std::size_t i = begin; i < end; ++i)
        {
            const std::size_t triangleIndex{
                m_triangleIndices[i]
            };

            const Triangle& triangle{
                m_triangle[triangleIndex]
            };

            for (std::size_t vertexIndex = 0;
                vertexIndex < 3;
                ++vertexIndex)
            {
                bounds.expand(
                    *triangle.vertex[vertexIndex].pos
                );
            }
        }

        return bounds;
    }


    [[nodiscard]] Eigen::Vector3d triangleCenter(
        const std::size_t triangleIndex) const
    {
        const Triangle& triangle{
            m_triangle[triangleIndex]
        };

        return (
            *triangle.vertex[0].pos +
            *triangle.vertex[1].pos +
            *triangle.vertex[2].pos
            ) / 3.0;
    }

    [[nodiscard]] static bool intersectTriangle(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        const Triangle& triangle,
        double& t,
        double& u,
        double& v) noexcept
    {
        return TriangularMesh::intersect(
            origin,
            direction,
            *triangle.vertex[0].pos,
            *triangle.vertex[1].pos,
            *triangle.vertex[2].pos,
            t,
            u,
            v);
    }
};

#endif // BVH_HPP