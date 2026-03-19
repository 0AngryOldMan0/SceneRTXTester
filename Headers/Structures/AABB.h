#pragma once

#include "Point3D.h"

struct AABB // Структура для хранения коробочки
{
    Vec3 v0, v1;
};

static_assert(sizeof(AABB) == 24); // Необходимое преобразование для сопоставления размера типа данных на ЦПУ и ГПУ