#pragma once

struct Vec3 // Точка
{
    float x, y, z;
};

static_assert(sizeof(Vec3) == 12); // Необходимое преобразование для сопоставления размера типа данных на ЦПУ и ГПУ