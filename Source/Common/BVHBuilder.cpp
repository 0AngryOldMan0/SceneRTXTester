#include "BVHBuilder.h"
#include "MathUtils.h"
#include <chrono>
#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

namespace
{
    // Быстрый доступ к центроиду коробки (centroid предвычислен, но иногда полезно иметь fallback
    inline int clampInt(int v, int lo, int hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}

BVHBuilder::BVHBuilder(Strategy strategy)
    : strategy_(strategy)
{
}

void BVHBuilder::buildForObject(SceneObject &object)
{
    if (object.getTriangles().empty())
    {
        return;
    }

    const auto startTime = std::chrono::high_resolution_clock::now();

    // Вычисляем AABB для треугольников объекта
    std::vector<AABB> objectAABBs = calculateTrianglesAABB(object.getTriangles());

    // Строим BVH в зависимости от выбранной стратегии
    std::vector<BVHNode> nodes;
    int rootIndex = -1;

    switch (strategy_)
    {
    case Strategy::BottomUp:
        rootIndex = buildBottomUp(objectAABBs, nodes);
        break;
    case Strategy::TopDown:
        rootIndex = buildTopDown(objectAABBs, nodes);
        break;
    case Strategy::SAH:
        rootIndex = buildSAH(objectAABBs, nodes);
        break;
    }

    // Обновляем объект
    object.setBVHNodes(nodes);
    object.setRootIndex(rootIndex);

    // Обновляем AABB треугольников
    auto &triangles = const_cast<std::vector<Triangle> &>(object.getTriangles());
    const size_t triCount = triangles.size();
    for (size_t i = 0; i < triCount; ++i)
    {
        triangles[i].ABoBa = objectAABBs[i];
    }

    const auto endTime = std::chrono::high_resolution_clock::now();

    lastStats_.buildTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    lastStats_.nodesCreated = nodes.size();
    lastStats_.leafNodes = static_cast<size_t>(std::count_if(nodes.begin(), nodes.end(),
                                                            [](const BVHNode &node)
                                                            { return node.tri >= 0; }));

    // Вычисляем глубину дерева (корректно и для top-down, и для bottom-up)
    std::vector<int> levels(nodes.size(), -1);
    calculateNodeLevels(nodes, levels, lastStats_.maxDepth);
}

void BVHBuilder::buildForScene(std::vector<SceneObject> &objects)
{
    // Пока просто строим BVH для каждого объекта отдельно
    // В будущем можно добавить построение общего BVH для сцены
    for (auto &obj : objects)
    {
        buildForObject(obj);
    }
}

std::vector<AABB> BVHBuilder::calculateTrianglesAABB(const std::vector<Triangle> &triangles) const
{
    std::vector<AABB> aabbs;
    aabbs.reserve(triangles.size());

    for (const auto &tri : triangles)
    {
        const float minx = std::min(std::min(tri.v0.x, tri.v1.x), tri.v2.x);
        const float miny = std::min(std::min(tri.v0.y, tri.v1.y), tri.v2.y);
        const float minz = std::min(std::min(tri.v0.z, tri.v1.z), tri.v2.z);

        const float maxx = std::max(std::max(tri.v0.x, tri.v1.x), tri.v2.x);
        const float maxy = std::max(std::max(tri.v0.y, tri.v1.y), tri.v2.y);
        const float maxz = std::max(std::max(tri.v0.z, tri.v1.z), tri.v2.z);

        aabbs.emplace_back(AABB{Vec3{minx, miny, minz}, Vec3{maxx, maxy, maxz}});
    }

    return aabbs;
}

int BVHBuilder::buildBottomUp(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes)
{
    outNodes.clear();
    if (boxes.empty())
    {
        return -1;
    }

    const int n = static_cast<int>(boxes.size());
    outNodes.reserve(2 * n - 1);

    // Active set + O(1) erase через таблицу позиций
    std::vector<int> active;
    active.reserve(n);

    // Максимум узлов известен заранее; posIndex[parentIdx] -> позиция в active
    std::vector<int> posIndex(static_cast<size_t>(2 * n - 1), -1);

    for (int i = 0; i < n; ++i)
    {
        BVHNode leaf;
        leaf.box = normalizeBox(boxes[i]);
        leaf.tri = i;
        leaf.left = -1;
        leaf.right = -1;
        leaf.parent = -1;

        outNodes.push_back(leaf);
        posIndex[static_cast<size_t>(i)] = static_cast<int>(active.size());
        active.push_back(i);
    }

    if (n == 1)
    {
        return active[0];
    }

    std::vector<int> chain;
    chain.reserve(static_cast<size_t>(std::min(n, 256)));

    auto eraseActiveFast = [&](int idx)
    {
        const int p = posIndex[static_cast<size_t>(idx)];
        if (p < 0)
            return;
        const int last = active.back();
        active[p] = last;
        posIndex[static_cast<size_t>(last)] = p;
        active.pop_back();
        posIndex[static_cast<size_t>(idx)] = -1;
    };

    while (active.size() > 1)
    {
        if (chain.empty())
        {
            chain.push_back(active[0]);
        }

        const int a = chain.back();
        int b = -1;
        float best = std::numeric_limits<float>::infinity();

        // Поиск ближайшего соседа по mergeCost (O(m))
        for (int k : active)
        {
            if (k == a)
                continue;
            const float c = mergeCost(outNodes[a].box, outNodes[k].box);
            if (c < best)
            {
                best = c;
                b = k;
            }
        }
        assert(b != -1);

        if (chain.size() >= 2 && chain[chain.size() - 2] == b)
        {
            const int i = a;
            const int j = b;

            BVHNode parent;
            parent.left = i;
            parent.right = j;
            parent.box = unite(outNodes[i].box, outNodes[j].box);
            parent.tri = -1;
            parent.parent = -1;

            outNodes.push_back(parent);
            const int parentIndex = static_cast<int>(outNodes.size()) - 1;

            outNodes[i].parent = parentIndex;
            outNodes[j].parent = parentIndex;

            eraseActiveFast(i);
            eraseActiveFast(j);

            // Добавляем нового родителя в active
            posIndex[static_cast<size_t>(parentIndex)] = static_cast<int>(active.size());
            active.push_back(parentIndex);

            chain.clear();
        }
        else
        {
            chain.push_back(b);
        }
    }

    return active[0];
}

// -------------------------------------------------------------
// Top-Down построение BVH (median split по центроидам)
// -------------------------------------------------------------

int BVHBuilder::buildTopDown(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes)
{
    outNodes.clear();
    if (boxes.empty())
    {
        return -1;
    }

    const int n = static_cast<int>(boxes.size());
    outNodes.reserve(2 * n - 1);

    // Предвычисляем центроиды всех AABB один раз (ускоряет split и SAH)
    std::vector<float> centerX(static_cast<size_t>(n));
    std::vector<float> centerY(static_cast<size_t>(n));
    std::vector<float> centerZ(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        centerX[static_cast<size_t>(i)] = 0.5f * (boxes[i].v0.x + boxes[i].v1.x);
        centerY[static_cast<size_t>(i)] = 0.5f * (boxes[i].v0.y + boxes[i].v1.y);
        centerZ[static_cast<size_t>(i)] = 0.5f * (boxes[i].v0.z + boxes[i].v1.z);
    }

    std::vector<int> primIndices(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        primIndices[static_cast<size_t>(i)] = i;

    return buildTopDownRecursive(boxes, centerX, centerY, centerZ, primIndices, 0, n, outNodes);
}

int BVHBuilder::buildTopDownRecursive(const std::vector<AABB> &boxes,
                                      const std::vector<float> &centerX,
                                      const std::vector<float> &centerY,
                                      const std::vector<float> &centerZ,
                                      std::vector<int> &primIndices,
                                      int begin,
                                      int end,
                                      std::vector<BVHNode> &outNodes)
{
    const int count = end - begin;
    if (count <= 0)
        return -1;

    // Bounding box узла + bounds центроидов (один проход)
    const int firstIdx = primIndices[begin];
    AABB nodeBox = boxes[firstIdx];

    float minCx = centerX[static_cast<size_t>(firstIdx)];
    float maxCx = minCx;
    float minCy = centerY[static_cast<size_t>(firstIdx)];
    float maxCy = minCy;
    float minCz = centerZ[static_cast<size_t>(firstIdx)];
    float maxCz = minCz;

    for (int i = begin + 1; i < end; ++i)
    {
        const int idx = primIndices[i];
        nodeBox = unite(nodeBox, boxes[idx]);

        const float cx = centerX[static_cast<size_t>(idx)];
        const float cy = centerY[static_cast<size_t>(idx)];
        const float cz = centerZ[static_cast<size_t>(idx)];

        if (cx < minCx)
            minCx = cx;
        if (cx > maxCx)
            maxCx = cx;
        if (cy < minCy)
            minCy = cy;
        if (cy > maxCy)
            maxCy = cy;
        if (cz < minCz)
            minCz = cz;
        if (cz > maxCz)
            maxCz = cz;
    }

    const int nodeIndex = static_cast<int>(outNodes.size());
    outNodes.emplace_back();
    BVHNode &node = outNodes.back();
    node.box = nodeBox;
    node.parent = -1;
    node.left = -1;
    node.right = -1;
    node.tri = -1;

    if (count == 1)
    {
        node.tri = firstIdx;
        return nodeIndex;
    }

    // Ось разбиения — по наибольшему разбросу центроидов (обычно лучше, чем по nodeBox.extent)
    const float ex = maxCx - minCx;
    const float ey = maxCy - minCy;
    const float ez = maxCz - minCz;

    int axis = 0;
    if (ey > ex && ey >= ez)
        axis = 1;
    else if (ez > ex && ez >= ey)
        axis = 2;

    const float *cPtr = nullptr;
    if (axis == 0)
        cPtr = centerX.data();
    else if (axis == 1)
        cPtr = centerY.data();
    else
        cPtr = centerZ.data();

    const int mid = begin + (count >> 1);
    auto itBegin = primIndices.begin() + begin;
    auto itMid = primIndices.begin() + mid;
    auto itEnd = primIndices.begin() + end;

    std::nth_element(itBegin, itMid, itEnd, [&](int a, int b)
                     { return cPtr[static_cast<size_t>(a)] < cPtr[static_cast<size_t>(b)]; });

    const int leftIndex = buildTopDownRecursive(boxes, centerX, centerY, centerZ, primIndices, begin, mid, outNodes);
    const int rightIndex = buildTopDownRecursive(boxes, centerX, centerY, centerZ, primIndices, mid, end, outNodes);

    node.left = leftIndex;
    node.right = rightIndex;

    if (leftIndex >= 0)
        outNodes[static_cast<size_t>(leftIndex)].parent = nodeIndex;
    if (rightIndex >= 0)
        outNodes[static_cast<size_t>(rightIndex)].parent = nodeIndex;

    return nodeIndex;
}

// -------------------------------------------------------------
// SAH-стратегия (binned SAH, быстрый и устойчивый)
// -------------------------------------------------------------

int BVHBuilder::buildSAH(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes)
{
    outNodes.clear();
    if (boxes.empty())
        return -1;

    const int n = static_cast<int>(boxes.size());
    outNodes.reserve(2 * n - 1);

    std::vector<float> centerX(static_cast<size_t>(n));
    std::vector<float> centerY(static_cast<size_t>(n));
    std::vector<float> centerZ(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        centerX[static_cast<size_t>(i)] = 0.5f * (boxes[i].v0.x + boxes[i].v1.x);
        centerY[static_cast<size_t>(i)] = 0.5f * (boxes[i].v0.y + boxes[i].v1.y);
        centerZ[static_cast<size_t>(i)] = 0.5f * (boxes[i].v0.z + boxes[i].v1.z);
    }

    std::vector<int> primIndices(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        primIndices[static_cast<size_t>(i)] = i;

    return buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, 0, n, outNodes);
}

int BVHBuilder::buildSAHRecursive(const std::vector<AABB> &boxes,
                                  const std::vector<float> &centerX,
                                  const std::vector<float> &centerY,
                                  const std::vector<float> &centerZ,
                                  std::vector<int> &primIndices,
                                  int begin,
                                  int end,
                                  std::vector<BVHNode> &outNodes)
{
    constexpr int kBins = 16;
    constexpr float kEps = 1e-6f;

    const int count = end - begin;
    if (count <= 0)
        return -1;

    const int firstIdx = primIndices[begin];
    AABB nodeBox = boxes[firstIdx];

    float minCx = centerX[static_cast<size_t>(firstIdx)];
    float maxCx = minCx;
    float minCy = centerY[static_cast<size_t>(firstIdx)];
    float maxCy = minCy;
    float minCz = centerZ[static_cast<size_t>(firstIdx)];
    float maxCz = minCz;

    for (int i = begin + 1; i < end; ++i)
    {
        const int idx = primIndices[i];
        nodeBox = unite(nodeBox, boxes[idx]);

        const float cx = centerX[static_cast<size_t>(idx)];
        const float cy = centerY[static_cast<size_t>(idx)];
        const float cz = centerZ[static_cast<size_t>(idx)];

        if (cx < minCx)
            minCx = cx;
        if (cx > maxCx)
            maxCx = cx;
        if (cy < minCy)
            minCy = cy;
        if (cy > maxCy)
            maxCy = cy;
        if (cz < minCz)
            minCz = cz;
        if (cz > maxCz)
            maxCz = cz;
    }

    const int nodeIndex = static_cast<int>(outNodes.size());
    outNodes.emplace_back();
    BVHNode &node = outNodes.back();
    node.box = nodeBox;
    node.parent = -1;
    node.left = -1;
    node.right = -1;
    node.tri = -1;

    if (count == 1)
    {
        node.tri = firstIdx;
        return nodeIndex;
    }

    // Деградация: если центроиды совпали, SAH бессмысленен -> median split
    const float ex = maxCx - minCx;
    const float ey = maxCy - minCy;
    const float ez = maxCz - minCz;
    if (ex < kEps && ey < kEps && ez < kEps)
    {
        // fallback median split по nodeBox extent (или просто по X)
        int axis = 0;
        const float bx = nodeBox.v1.x - nodeBox.v0.x;
        const float by = nodeBox.v1.y - nodeBox.v0.y;
        const float bz = nodeBox.v1.z - nodeBox.v0.z;
        if (by > bx && by >= bz)
            axis = 1;
        else if (bz > bx && bz >= by)
            axis = 2;

        const float *cPtr = (axis == 0) ? centerX.data() : (axis == 1) ? centerY.data()
                                                                        : centerZ.data();

        const int mid = begin + (count >> 1);
        auto itBegin = primIndices.begin() + begin;
        auto itMid = primIndices.begin() + mid;
        auto itEnd = primIndices.begin() + end;
        std::nth_element(itBegin, itMid, itEnd, [&](int a, int b)
                         { return cPtr[static_cast<size_t>(a)] < cPtr[static_cast<size_t>(b)]; });

        const int leftIndex = buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, begin, mid, outNodes);
        const int rightIndex = buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, mid, end, outNodes);

        node.left = leftIndex;
        node.right = rightIndex;
        if (leftIndex >= 0)
            outNodes[static_cast<size_t>(leftIndex)].parent = nodeIndex;
        if (rightIndex >= 0)
            outNodes[static_cast<size_t>(rightIndex)].parent = nodeIndex;
        return nodeIndex;
    }

    // SAH: выбираем лучшую ось и лучший split по бинам
    float bestCost = std::numeric_limits<float>::infinity();
    int bestAxis = -1;
    int bestSplitBin = -1;
    float bestMinC = 0.0f;
    float bestInvExtent = 0.0f;

    auto evalAxis = [&](int axis, float cmin, float cmax)
    {
        if (cmax - cmin < kEps)
            return;

        // scale: (kBins / extent)
        const float invExtent = static_cast<float>(kBins) / (cmax - cmin);

        int binCount[kBins] = {0};
        AABB binBox[kBins];
        bool binValid[kBins] = {false};

        const float *cPtr = (axis == 0) ? centerX.data() : (axis == 1) ? centerY.data()
                                                                        : centerZ.data();

        for (int i = begin; i < end; ++i)
        {
            const int idx = primIndices[i];
            const float c = cPtr[static_cast<size_t>(idx)];
            int b = static_cast<int>((c - cmin) * invExtent);
            b = clampInt(b, 0, kBins - 1);

            binCount[b] += 1;
            if (!binValid[b])
            {
                binBox[b] = boxes[idx];
                binValid[b] = true;
            }
            else
            {
                binBox[b] = unite(binBox[b], boxes[idx]);
            }
        }

        // Prefix
        int leftCount[kBins] = {0};
        AABB leftBox[kBins];
        bool leftValid[kBins] = {false};

        int accCount = 0;
        for (int i = 0; i < kBins; ++i)
        {
            if (binCount[i] > 0)
            {
                if (!leftValid[i] && accCount == 0)
                {
                    leftBox[i] = binBox[i];
                }
                else if (accCount > 0)
                {
                    leftBox[i] = unite(leftBox[i - 1], binBox[i]);
                }
                else
                {
                    leftBox[i] = binBox[i];
                }
                accCount += binCount[i];
                leftCount[i] = accCount;
                leftValid[i] = true;
            }
            else
            {
                if (i > 0 && leftValid[i - 1])
                {
                    leftBox[i] = leftBox[i - 1];
                    leftCount[i] = leftCount[i - 1];
                    leftValid[i] = true;
                }
            }
        }

        // Suffix
        int rightCount[kBins] = {0};
        AABB rightBox[kBins];
        bool rightValid[kBins] = {false};

        accCount = 0;
        for (int i = kBins - 1; i >= 0; --i)
        {
            if (binCount[i] > 0)
            {
                if (!rightValid[i] && accCount == 0)
                {
                    rightBox[i] = binBox[i];
                }
                else if (accCount > 0)
                {
                    rightBox[i] = unite(rightBox[i + 1], binBox[i]);
                }
                else
                {
                    rightBox[i] = binBox[i];
                }
                accCount += binCount[i];
                rightCount[i] = accCount;
                rightValid[i] = true;
            }
            else
            {
                if (i < kBins - 1 && rightValid[i + 1])
                {
                    rightBox[i] = rightBox[i + 1];
                    rightCount[i] = rightCount[i + 1];
                    rightValid[i] = true;
                }
            }
        }

        for (int split = 0; split < kBins - 1; ++split)
        {
            const int lc = leftCount[split];
            const int rc = rightCount[split + 1];
            if (lc == 0 || rc == 0)
                continue;

            const float cost = surfaceArea(leftBox[split]) * static_cast<float>(lc) +
                               surfaceArea(rightBox[split + 1]) * static_cast<float>(rc);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestAxis = axis;
                bestSplitBin = split;
                bestMinC = cmin;
                bestInvExtent = invExtent;
            }
        }
    };

    evalAxis(0, minCx, maxCx);
    evalAxis(1, minCy, maxCy);
    evalAxis(2, minCz, maxCz);

    // Если SAH не смог найти split (редко), fallback median
    if (bestAxis < 0)
    {
        int axis = 0;
        if (ey > ex && ey >= ez)
            axis = 1;
        else if (ez > ex && ez >= ey)
            axis = 2;

        const float *cPtr = (axis == 0) ? centerX.data() : (axis == 1) ? centerY.data()
                                                                        : centerZ.data();

        const int mid = begin + (count >> 1);
        auto itBegin = primIndices.begin() + begin;
        auto itMid = primIndices.begin() + mid;
        auto itEnd = primIndices.begin() + end;
        std::nth_element(itBegin, itMid, itEnd, [&](int a, int b)
                         { return cPtr[static_cast<size_t>(a)] < cPtr[static_cast<size_t>(b)]; });

        const int leftIndex = buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, begin, mid, outNodes);
        const int rightIndex = buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, mid, end, outNodes);

        node.left = leftIndex;
        node.right = rightIndex;
        if (leftIndex >= 0)
            outNodes[static_cast<size_t>(leftIndex)].parent = nodeIndex;
        if (rightIndex >= 0)
            outNodes[static_cast<size_t>(rightIndex)].parent = nodeIndex;
        return nodeIndex;
    }

    const float *cPtr = (bestAxis == 0) ? centerX.data() : (bestAxis == 1) ? centerY.data()
                                                                           : centerZ.data();

    // Partition по bins <= bestSplitBin
    auto itBegin = primIndices.begin() + begin;
    auto itEnd = primIndices.begin() + end;

    auto midIt = std::partition(itBegin, itEnd, [&](int idx)
                                {
        const float c = cPtr[static_cast<size_t>(idx)];
        int b = static_cast<int>((c - bestMinC) * bestInvExtent);
        b = clampInt(b, 0, kBins - 1);
        return b <= bestSplitBin; });

    int mid = static_cast<int>(std::distance(primIndices.begin(), midIt));

    // На случай, если split оказался пустым — fallback median
    if (mid == begin || mid == end)
    {
        mid = begin + (count >> 1);
        auto itMid = primIndices.begin() + mid;
        std::nth_element(itBegin, itMid, itEnd, [&](int a, int b)
                         { return cPtr[static_cast<size_t>(a)] < cPtr[static_cast<size_t>(b)]; });
    }

    const int leftIndex = buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, begin, mid, outNodes);
    const int rightIndex = buildSAHRecursive(boxes, centerX, centerY, centerZ, primIndices, mid, end, outNodes);

    node.left = leftIndex;
    node.right = rightIndex;
    if (leftIndex >= 0)
        outNodes[static_cast<size_t>(leftIndex)].parent = nodeIndex;
    if (rightIndex >= 0)
        outNodes[static_cast<size_t>(rightIndex)].parent = nodeIndex;

    return nodeIndex;
}

void BVHBuilder::eraseActive(std::vector<int> &active, int idx) const
{
    // (Сохранено для совместимости; основной код использует eraseActiveFast в buildBottomUp)
    for (int t = 0; t < static_cast<int>(active.size()); ++t)
    {
        if (active[t] == idx)
        {
            active[t] = active.back();
            active.pop_back();
            return;
        }
    }
}

void BVHBuilder::calculateNodeLevels(const std::vector<BVHNode> &nodes,
                                     std::vector<int> &levels,
                                     size_t &maxLevel) const
{
    maxLevel = 0;
    if (nodes.empty())
        return;

    std::fill(levels.begin(), levels.end(), -1);

    // Ищем корни (parent < 0). Обычно корень один.
    std::vector<int> roots;
    roots.reserve(4);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (nodes[i].parent < 0)
            roots.push_back(static_cast<int>(i));
    }
    if (roots.empty())
        roots.push_back(0);

    // DFS/stack: (nodeIndex, depth)
    std::vector<std::pair<int, int>> stack;
    stack.reserve(nodes.size());

    for (int r : roots)
        stack.emplace_back(r, 0);

    while (!stack.empty())
    {
        const auto [ni, depth] = stack.back();
        stack.pop_back();

        if (ni < 0 || static_cast<size_t>(ni) >= nodes.size())
            continue;

        const size_t u = static_cast<size_t>(ni);
        if (levels[u] >= depth)
            continue;

        levels[u] = depth;
        if (static_cast<size_t>(depth) > maxLevel)
            maxLevel = static_cast<size_t>(depth);

        const BVHNode &n = nodes[u];
        if (n.left >= 0)
            stack.emplace_back(n.left, depth + 1);
        if (n.right >= 0)
            stack.emplace_back(n.right, depth + 1);
    }
}

// -------------------------------------------------------------
// Универсальный вход: выбирает стратегию по strategy_
// -------------------------------------------------------------

int BVHBuilder::buildFromAABBs(const std::vector<AABB> &boxes, std::vector<BVHNode> &outNodes)
{
    switch (strategy_)
    {
    case Strategy::TopDown:
        return buildTopDown(boxes, outNodes);
    case Strategy::SAH:
        return buildSAH(boxes, outNodes);
    case Strategy::BottomUp:
    default:
        return buildBottomUp(boxes, outNodes);
    }
}
