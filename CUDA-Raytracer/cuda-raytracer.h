#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#include "cuda_runtime.h"
#include "cuda.h"
#include <device_launch_parameters.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Utility Includes
#include "vec3.h"
#include "interval.h"

#endif // CUDA_RAYTRACER_H
