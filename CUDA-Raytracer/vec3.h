#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include "cuda_runtime.h" 

class vec3 {
public:
	__host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator *=(const vec3& v) {
		e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;

    }

    __host__ __device__ vec3& operator *=(const float t) {
		e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;

    }

    __host__ __device__ vec3& operator /=(const float t) {
        return *this *= (1/t);
    }

	__host__ __device__ float length_squared() const {
        return dot(*this, *this);
    }

    __host__ __device__ float length() const {
        return std::sqrt(length_squared());
    }

    
private:
    float e[3];
};

using point3 = vec3;

// Inline Functions
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

__host__ __device__ inline float dot(const vec3& v, const vec3& u) {
    return v.x() * u.x() + v.y() * u.y() + v.z() * u.z();
}

__host__ __device__ inline vec3 cross(const vec3& v, const vec3& u) {
    return vec3(
            v.y()*u.z() - v.z()*u.y(), 
            v.z()*u.x() - v.x()*u.z(), 
            v.x()*u.y() - v.y()*u.x());
}

__host__ __device__ inline vec3 operator+(const vec3& v, const vec3& u) {
    return vec3(v.x() + u.x(), v.y() + u.y(), v.z() + u.z());
}

__host__ __device__ inline vec3 operator*(const vec3& v, const vec3& u) {
    return vec3(v.x() * u.x(), v.y() * u.y(), v.z() * u.z());
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return vec3(v.x() * t, v.y() * t, v.z() * t);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(v.x() * t, v.y() * t, v.z() * t);
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif // VEC3_H

