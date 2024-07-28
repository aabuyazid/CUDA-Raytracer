#ifndef CAMERA_H
#define CAMERA_H

#include "cuda-raytracer.h"

class camera {
public: 
	int img_width, img_height;
	double viewport_height, vfov;

	__host__ camera();
	__host__ camera(int img_width, int img_height, double viewport_height, double vfov);
	__host__ ~camera() = default;

	point3 look_from = point3(0, 0, 0);
	point3 look_at   = point3(0, 0, -1);
	vec3   vup       = vec3(0, 1, 0);

	__host__ void test_render();

private:
	size_t fb_size;
	float* ft_buf;
	int tx = 8, ty = 8;

	// __host__ void test_initialize();

	__global__ void test_render_kernel(float* fb, int max_x, int max_y);
};

#endif // CAMERA_H

