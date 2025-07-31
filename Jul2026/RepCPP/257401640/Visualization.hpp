#pragma once
#include <vector>
#include "Shapes.hpp"
#include "Collision.hpp"

using DrawCall = std::vector<std::unique_ptr<ShapeHelper>>;
using DrawStack = std::vector<DrawCall>;

class Visualization
{
public:
	virtual void simulate(const Hull& a, const Hull& b) = 0;

	DrawStack& operator()() { return mDrawStack; }
	DrawCall& operator[](size_t i) { return mDrawStack[i]; }
	size_t size() const { return mDrawStack.size(); } 

protected:
	DrawStack mDrawStack;
};
