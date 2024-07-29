#include "camera.h"

// Camera Function Implementations Start
__host__ void camera::initialize() {
    int num_pixels = img_width * img_height;
    ib_size = num_pixels*sizeof(vec3);
    
    // allocate FB
    checkCudaErrors(cudaMallocManaged((void **)&img_buf, ib_size));
}
__host__ camera::camera() : img_width(512), img_height(512), 
viewport_height(2.0), vfov(90.0) {
    initialize();
}

__host__ camera::camera(int img_width, int img_height, double viewport_height, double vfov) : 
img_width(img_width), img_height(img_height), viewport_height(viewport_height), vfov(vfov) {
    initialize();
}

__host__ std::shared_ptr<image> camera::create_frame() {
    auto frame = std::make_shared<image>(img_width, img_height);
    for (int y = 0; y < img_height; y++) {
        for (int x = 0; x < img_width; x++) {
            size_t pixel_idx = (y * img_width) + x;
            frame->setPixel(x, y, img_buf[pixel_idx]);
        }
    }
    return frame;
}

__host__ std::shared_ptr<image> camera::gradient_render() {
    dim3 blocks(img_width / tx + 1, img_height / ty + 1);
    dim3 threads(tx, ty);
    gradient_render_kernel<<<blocks, threads>>>(img_buf, img_width, img_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return create_frame();
}

// Camera Function Implementations End

// GPU Kernel/Functions Start
__global__ void gradient_render_kernel(color* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    fb[pixel_index] = color(float(i) / max_x, float(j) / max_y, 0.2);
}

__global__ void skybox_render_kernel(color* fb, int img_width, int img_height,
point3 origin, point3 pixel00_loc, vec3 delta_u, vec3 delta_h) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= img_width) || (j >= img_height)) return;
   int pixel_index = j*img_width + i;
   float u = float(i) / float(img_width);
   float v = float(j) / float(img_height);
   ray r(origin, pixel00_loc + u*delta_u + v*delta_h);
   fb[pixel_index] = ray_color(r);

}

__device__ vec3 ray_color(const ray& r) {
	vec3 unit_direction = unit_vector(r.direction());
	float a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}
// GPU Kernel/Functions End
