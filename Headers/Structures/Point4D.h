#pragma once

struct Vec4
{
    float x, y, z, w;
};

static_assert(sizeof(Vec4) == 16);
