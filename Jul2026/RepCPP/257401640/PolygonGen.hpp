#pragma once
#include <glm/glm.hpp>
#include <random>
#include <vector>
#include <span>

#include "VertexPool.hpp"

class PolygonGen
{
public:
	using PolyArray = std::vector<std::span<glm::vec2>>;
public:
	PolygonGen(size_t memSize = 1024 * 1024); // ~1M vertices (4MB memory)

	void setPolygonArea(glm::vec2 areaMin, glm::vec2 areaMax);
	void setPolygonSize(float size);
	void setMaxVertices(size_t size);

	PolyArray& generatePolygons(size_t count);
	PolyArray& resizePolygons(size_t count);
	PolyArray& regenerateLastPolygons(size_t count);
	std::vector<glm::vec2>& getDirections();

	std::span<glm::vec2> operator[](size_t i) const;
	
	PolyArray& data();
	size_t size() const;

private:
	std::span<glm::vec2> generatePolygon();

private:
	VertexPool mPool;
	//VertexPool mDirPool;
	PolyArray mPolygons;
	std::vector<glm::vec2> mDirections;

	glm::vec2 mAreaMin;
	glm::vec2 mAreaMax;

	float mMaxPolygonSize;
	size_t mMaxVertices;

	std::random_device mRandomDevice;
	std::mt19937 mGenerator;

	std::uniform_real_distribution<float> mDistDir{ -1.f, 1.f };
	std::uniform_real_distribution<float> mDistAngle{ 0.f, 2 * 3.14159265358979f };
	std::uniform_real_distribution<float> mDistX;
	std::uniform_real_distribution<float> mDistY;
	std::uniform_real_distribution<float> mDistPolySize;
	std::uniform_int_distribution<size_t> mDistMaxVert;
};