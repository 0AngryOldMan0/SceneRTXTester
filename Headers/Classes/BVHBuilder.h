#pragma once

#include "BVHNode.h"
#include "Triangles.h"
#include "AABB.h"
#include "SceneObject.h"
#include <vector>
#include <memory>

class BVHBuilder
{
public:
    enum class Strategy
    {
        BottomUp, // текущий алгоритм «цепочка ближайших соседей»
        TopDown,  // рекурсивный «сверху-вниз»
        SAH       // Surface Area Heuristic (binned SAH)
    };

    BVHBuilder(Strategy strategy = Strategy::TopDown);

    // Основные методы построения
    void buildForObject(SceneObject &object);
    void buildForScene(std::vector<SceneObject> &objects);

    // Геттеры/сеттеры
    Strategy getStrategy() const { return strategy_; }
    void setStrategy(Strategy strategy) { strategy_ = strategy; }

    // Статистика
    struct BuildStats
    {
        size_t nodesCreated = 0;
        size_t leafNodes = 0;
        size_t maxDepth = 0;
        double buildTimeMs = 0.0;
    };

    const BuildStats &getLastBuildStats() const { return lastStats_; }

    // Универсальный вход из «внешнего мира»:
    // здесь уже учитывается выбранная стратегия (BottomUp / TopDown / SAH)
    int buildFromAABBs(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes);

private:
    Strategy strategy_;
    BuildStats lastStats_;

    // Вспомогательные методы
    std::vector<AABB> calculateTrianglesAABB(const std::vector<Triangle> &triangles) const;

    // Стратегии построения
    int buildBottomUp(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes);
    int buildTopDown(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes);
    int buildSAH(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes);

    // Внутренние рекурсивные помощники (centroids предвычислены один раз)
    int buildTopDownRecursive(const std::vector<AABB> &boxes,
                              const std::vector<float> &centerX,
                              const std::vector<float> &centerY,
                              const std::vector<float> &centerZ,
                              std::vector<int> &primIndices,
                              int begin,
                              int end,
                              std::vector<BVHNode> &outNodes);

    int buildSAHRecursive(const std::vector<AABB> &boxes,
                          const std::vector<float> &centerX,
                          const std::vector<float> &centerY,
                          const std::vector<float> &centerZ,
                          std::vector<int> &primIndices,
                          int begin,
                          int end,
                          std::vector<BVHNode> &outNodes);

    // Вспомогательные функции из старого main.cpp
    void eraseActive(std::vector<int> &active, int idx) const;

    void calculateNodeLevels(const std::vector<BVHNode> &nodes,
                             std::vector<int> &levels,
                             size_t &maxLevel) const;
};
