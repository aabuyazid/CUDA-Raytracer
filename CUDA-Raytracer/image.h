#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <string>
#include <vector>

#include "vec3.h"

using color = vec3;

class image
{
public:
	image(int width, int height);
	virtual ~image();
	void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
    void setPixel(int x, int y, const color& pixelColor);
	void writeToFile(const std::string &filename);
	int getWidth() const { return width; }
	int getHeight() const { return height; }

private:
	int width;
	int height;
	int comp;
	std::vector<unsigned char> pixels;
};

#endif
