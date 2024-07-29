#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#include "cuda_runtime.h"
#include "cuda.h"
#include <device_launch_parameters.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

// Utility Includes
#include "vec3.h"
#include "interval.h"
#include "ray.h"

// Constants

const float pi = 3.1415926535897932385f;

// Utility Functions
float degrees_to_radians(float degrees);

#endif // CUDA_RAYTRACER_H
