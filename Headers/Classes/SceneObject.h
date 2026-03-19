#pragma once

#include "Triangles.h"
#include "BVHNode.h"

#include <string>
#include <vector>
#include <unordered_map>

class SceneObject
{
public:
    SceneObject() = default;
    explicit SceneObject(const std::string &name);

    // ---- Геттеры ----
    const std::string &getName() const;
    const std::vector<Triangle> &getTriangles() const;
    std::vector<Triangle> &getTrianglesMutable(); // чтобы не делать const_cast в meta loader
    const std::vector<BVHNode> &getBVHNodes() const;
    int getRootIndex() const;

    // Вернёт nullptr, если для этого треугольника нет материала
    const std::string *getTriangleMaterialName(std::size_t triIndex) const;

    // Зарегистрировать имя материала и получить его id (0..N-1), либо вернуть уже существующий
    int registerMaterialName(const std::string &matName);

    // ---- Сеттеры / модификация ----
    void setName(const std::string &newName);

    // Устанавливает треугольники и сбрасывает привязку материалов (все tri -> -1)
    void setTriangles(const std::vector<Triangle> &tris);
    void setTriangles(std::vector<Triangle> &&tris); // move

    void setBVHNodes(const std::vector<BVHNode> &newNodes);
    void setBVHNodes(std::vector<BVHNode> &&newNodes); // move

    void setRootIndex(int index);

    // Добавление треугольника без материала (materialId = -1)
    void addTriangle(const Triangle &tri);

    // Добавление треугольника с известным materialId
    void addTriangle(const Triangle &tri, int materialId);

    // Добавление треугольника по имени материала (вызовет registerMaterialName)
    void addTriangle(const Triangle &tri, const std::string &materialName);

    void addBVHNode(const BVHNode &node);
    void computeTriangleAABBs();

    // ---- Вспомогательные методы ----
    bool isEmpty() const;
    std::size_t getTriangleCount() const;

    bool hasBVH() const;

    const BVHNode &getBVHNode(int index) const;
    const BVHNode &getRootNode() const;

    const Vec3 &getBaseColor() const;
    void setBaseColor(const Vec3 &color);

    // Применить один материал ко всему объекту (fallback / старый режим)
    void applyMaterial(const Vec3 &baseColor,
                       const Vec3 &emission,
                       float metallic,
                       float roughness);

    void clear();

private:
    friend class BVHBuilder;

    std::string name;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    int rootIndex = -1;

    Vec3 baseColor_{1.0f, 1.0f, 1.0f};

    // ---- Многоматериальность ----
    std::vector<std::string> materialNames_;
    std::vector<int> triangleMaterialIds_;

    // быстрый поиск id по имени
    std::unordered_map<std::string, int> materialNameToId_;
};
