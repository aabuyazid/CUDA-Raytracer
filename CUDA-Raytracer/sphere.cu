#include "sphere.h"

// Hit Record
__device__ void hit_record::set_face_normal(const ray& r, const vec3& outward_normal) {
	front_face = dot(r.direction(), outward_normal) < 0.0f;
	normal = front_face ? outward_normal : -outward_normal;
}

// Sphere
__device__ hit_record sphere::hit(const ray& r, interval ray_t) {
	hit_record rec;

	vec3 oc = center - r.origin();
	float a = r.direction().length_squared();
	float h = dot(r.direction(), oc);
	float c = oc.length_squared() - (radius * radius);
	float discriminant = (h*h) - (a*c);

	if (discriminant < 0.0f) {
		return rec;
	}

	float sqrtd = std::sqrt(discriminant);

	// Find the nearest root that lies in teh acceptable range
	float root = (h - sqrtd) / a;
	if (!ray_t.surrounds(root)) {
		root = (h + sqrtd) / a;
		if (!ray_t.surrounds(root)) {
			return rec;
		}
	}

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius; // Normal vector of intersection of point and center of a sphere
    rec.set_face_normal(r, outward_normal);

    return rec;
}
