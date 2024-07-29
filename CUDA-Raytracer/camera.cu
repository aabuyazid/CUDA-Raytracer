#include "camera.h"

// Camera Function Implementations Start
__host__ void camera::initialize() {
    // Camera Set-up
    float focal_length = 1.0;
    float aspect_ratio = img_width / img_height;
    float theta = degrees_to_radians(vfov);
    float h = std::tan(theta/2);
    viewport_height *= (h * focal_length);
    viewport_width = viewport_height * aspect_ratio;

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame
    w = unit_vector(look_from - look_at);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Horizontal vector u and vertical vector v of viewport edges
    vec3 viewport_u = viewport_width * u;
    vec3 viewport_v = viewport_height * -v;

    delta_u = viewport_u / img_width;
    delta_v = viewport_v / img_height;

    // Location of the upper left pixel
    vec3 viewport_upper_left = look_from - (focal_length * w) - viewport_u/2 - viewport_v/2;
    pixel00_loc = viewport_upper_left + 0.5f * (delta_u + delta_v);

    // Frame Buffer Set-up
    int num_pixels = img_width * img_height;
    ib_size = num_pixels*sizeof(vec3);

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

__host__ std::shared_ptr<image> camera::skybox_render() {
    dim3 blocks(img_width / tx + 1, img_height / ty + 1);
    dim3 threads(tx, ty);

    skybox_render_kernel<<<blocks, threads >>>(img_buf, img_width, img_height,
        look_from, pixel00_loc, delta_u, delta_v);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return create_frame();
}

__host__ std::shared_ptr<image> camera::two_sphere_render(sphere** d_sphere_list, int num_spheres) {
    dim3 blocks(img_width / tx + 1, img_height / ty + 1, 1);
    dim3 threads(tx, ty, num_spheres);

    two_sphere_render_kernel<<<blocks, threads, img_width*img_height*num_spheres*sizeof(hit_record)>>>(img_buf, img_width, img_height,
        look_from, pixel00_loc, delta_u, delta_v, d_sphere_list, num_spheres);

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
point3 origin, point3 pixel00_loc, vec3 delta_u, vec3 delta_v) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   if((x >= img_width) || (y >= img_height)) return;
   int pixel_index = y*img_width + x;
   // float u = float(i) / float(img_width);
   // float v = float(j) / float(img_height);
   ray r(origin, pixel00_loc + x*delta_u + y*delta_v);
   fb[pixel_index] = ray_color(r);
}

__global__ void two_sphere_render_kernel(color* fb, int img_width, int img_height,
point3 origin, point3 pixel00_loc, vec3 delta_u, vec3 delta_v, sphere** d_sphere_list, int num_spheres) {
    extern __shared__ hit_record recs[];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if((x >= img_width) || (y >= img_height) || (z >= num_spheres)) return;
    int rec_idx = (y * img_width) + (x * num_spheres) + z;
    ray r(origin, pixel00_loc + x * delta_u + y * delta_v);
    recs[rec_idx] = (*d_sphere_list + z)->hit(r, interval(0.001f, +infinity));
    __syncthreads();
    if (!(x == 0 && y == 0 && z == 0)) return;

}

__device__ vec3 ray_color(const ray& r) {
	vec3 unit_direction = unit_vector(r.direction());
	float a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + (a * color(0.5f, 0.7f, 1.0f));
}
// GPU Kernel/Functions End
