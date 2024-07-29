#ifndef CAMERA_H
#define CAMERA_H

#include "cuda-raytracer.h"
#include "image.h"

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

	__host__ std::shared_ptr<image> gradient_render();

private:
	size_t ib_size;
	color* img_buf;
	int tx = 8, ty = 8;

	__host__ std::shared_ptr<image> create_frame();

	__host__ void initialize();

	// __host__ void test_initialize();
};

// GPU Kernel/Functions
__global__ void gradient_render_kernel(color* fb, int max_x, int max_y);
__global__ void skybox_render_kernel(color* fb, int img_width, int img_height,
point3 origin, point3 pixel00_loc, vec3 delta_u, vec3 delta_h);

__device__ vec3 ray_color(const ray& r);

#endif // CAMERA_H

