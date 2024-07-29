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
	__host__ __device__ inline float interval::size() const { return max - min; }
	__host__ __device__ inline bool interval::contains(float x) const {
		return min <= x && x <= max;
	}
	__host__ __device__ inline bool interval::surrounds(float x) const {
		return min < x && x < max;
	}
	
	__host__ __device__ inline float interval::clamp(float x) const {
		if (x < min) {
			return min;
		}
		if (x > max) {
			return max;
		}
		return x;
	}

	static const interval empty;
	static const interval universe;
};

#endif // INTERVAL_H

