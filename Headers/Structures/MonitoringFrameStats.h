#pragma once

#include <cstdint>

struct MonitoringFrameStats // ДОБАВЛЕНО: метрики одного кадра
{
    int frameIndex = 0;             // номер кадра
    double frameRenderTimeMs = 0.0; // время рендера (без сохранения PNG)
    double frameSaveTimeMs = 0.0;   // время записи PNG
    double frameTotalTimeMs = 0.0;  // суммарное время (рендер + сохранение)

    std::uint64_t raysTraced = 0;        // всего лучей (камерных) в кадре
    std::uint64_t raysHit = 0;           // лучей, попавших хотя бы в один треугольник
    std::uint64_t totalVisitedNodes = 0; // суммарное количество посещённых узлов BVH

    double avgRayTimeNs = 0.0;          // среднее время обработки одного луча (наносекунды)
    double avgVisitedNodesPerRay = 0.0; // среднее число посещённых узлов BVH на один луч
    double raysPerSecond = 0.0;         // оценка: лучей в секунду по времени рендера кадра
};