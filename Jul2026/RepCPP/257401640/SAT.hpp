#pragma once
#include <glm/glm.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <span>

#include "Visualization.hpp"
#include "Collision.hpp"

class SAT : public NarrowPhaseDetector
{
public:
	explicit SAT(const Hull& a, const Hull& b);
	virtual operator bool() override;

protected:
	struct MinMaxResult
	{
		float min;
		size_t minIndex;
		float max;
		size_t maxIndex;
	};

	std::vector<glm::vec2> getNormals(const Hull& hull);
	MinMaxResult getMinMax(const Hull& hull, const glm::vec2& axis);
};

class SATVisualizer : public Visualization, protected SAT
{
public:
	SATVisualizer();
	void simulate(const Hull& a, const Hull& b) override;
};
