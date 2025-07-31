#pragma once
#include <glm/glm.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <span>

#include "Visualization.hpp"
#include "Collision.hpp"

class GJK : public NarrowPhaseDetector
{
public:
	explicit GJK(const Hull& a, const Hull& b);
	virtual operator bool() override;

protected:
	glm::vec2 support(const glm::vec2& direction);

};

class GJKVisualizer : public Visualization, protected GJK 
{
public:
	GJKVisualizer();
	void simulate(const Hull& a, const Hull& b) override;
};