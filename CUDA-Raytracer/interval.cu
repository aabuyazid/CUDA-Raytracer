#include "interval.h"

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

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);
