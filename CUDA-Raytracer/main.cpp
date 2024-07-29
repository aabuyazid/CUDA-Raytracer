#include "cuda-raytracer.h"
#include "camera.h"

#include <iostream>

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

    auto frame = cam.test_render();

    frame->writeToFile(out_file);

    std::cout << "Render Complete\n";

    return 0;
}