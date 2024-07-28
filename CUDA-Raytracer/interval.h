#ifndef INTERVAL_H
#define INTERVAL_H

#include "cuda_runtime.h"
#include <limits>

const float infinity = std::numeric_limits<float>::infinity();

class interval {
public:
	float min, max;
	__host__ __device__ interval() : min(+infinity), max(-infinity) {}
	__host__ __device__ interval(float min, float max) : min(min), max(max) {}
	__host__ __device__ float size() const;
	__host__ __device__ bool contains(float x) const;
	__host__ __device__ bool surrounds(float x) const;

	__host__ __device__ float clamp(float x) const;

	static const interval empty;
	static const interval universe;
};

#endif // INTERVAL_H

