#ifndef SPHERE_H
#define SPHERE_H

#include "cuda-raytracer.h"

class hit_record {
public:
	point3 p;
	vec3 normal;
	double t;
	bool front_face;
	bool hit = false;

	__device__ hit_record() = default;

	__device__ void set_face_normal(const ray& r, const vec3& outward_normal);
};

class sphere {
public:
	__device__ sphere(const point3& center, float radius) : center(center), radius(radius) {}

	__device__ hit_record hit(const ray& r, interval ray_t);
private:
	point3 center;
	float radius;
};


#endif // SPHERE_H
