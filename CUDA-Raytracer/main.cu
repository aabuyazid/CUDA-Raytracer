#include "cuda-raytracer.h"
#include "camera.h"
#include "sphere.h"

#include <iostream>

// GPU Kernel Declarations
__global__ void create_two_spheres(sphere** d_list);

camera initialize_camera(int argc, char* argv[]) {
    if (argc == 2) {
        return camera();
    }

    int img_width  = atoi(argv[2]);
    int img_height = atoi(argv[3]);
    double viewport_height = atof(argv[4]);
    double vfov = atof(argv[5]);

    // Camera
    return camera(img_width, img_height, viewport_height, vfov);


}

int main(int argc, char* argv[]) {
    std::cout << argc << "\n";
    if (argc != 2 && argc != 6) {
        std::cout << "Usage: ./cuda-raytracer <output file> <img width> <img height> <viewport height> <fov> \n";
        std::cout << "Usage: ./cuda-raytracer <output file>  \n";
        return -1;
    }
    
    // Argument Reading
    std::string out_file = argv[1];

    camera cam = initialize_camera(argc, argv);
    
    // Populating World
    sphere** d_sphere_list;
    int num_spheres = 2;
    checkCudaErrors(cudaMalloc((void***)&d_sphere_list, num_spheres * sizeof(sphere*)));
    
    create_two_spheres<<<1, 1>>>(d_sphere_list);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render the world

    // auto frame = cam.gradient_render();
    auto frame = cam.skybox_render();

    frame->writeToFile(out_file);

    std::cout << "Render Complete\n";

    return 0;
}

// GPU Kernels
__global__ void create_two_spheres(sphere** d_list) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100); 
    }
}