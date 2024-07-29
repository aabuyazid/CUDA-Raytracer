#ifndef RAY_H
#define RAY_H

#include "cuda-raytracer.h"

class ray {
public:
	__device__ ray(){}
	__device__ ray(const point3& orig, const vec3& dir) : orig(orig), dir(dir) {}

	__device__ const point3& origin()  const { return orig; }
	__device__ const vec3& direction() const { return dir; }

	__device__ point3 at(double t) const { return orig + t * dir; }


private:
	point3 orig;
	vec3 dir;
};

#endif // RAY_H

