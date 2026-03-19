#pragma once

#include "AABB.h"

struct BVHNode // Структура для хранения характеристик ноды
{
    AABB box;
    int left = -1;
    int right = -1;
    int parent = -1;
    int tri = -1;
};

static_assert(sizeof(BVHNode) == 40); // Необходимое преобразование для сопоставления размера типа данных на ЦПУ и ГПУ