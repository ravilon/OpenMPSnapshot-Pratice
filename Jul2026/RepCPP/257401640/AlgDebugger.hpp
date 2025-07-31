#pragma once
#include <glm/glm.hpp>
#include <SFML/Graphics.hpp>
#include <memory>

#include "Level.hpp"
#include "Visualization.hpp"
#include "PolygonGen.hpp"
#include "Collision.hpp"


class AlgDebugger : public Level
{
public:
	AlgDebugger();
	virtual ~AlgDebugger() = default;

	virtual void update(float dt) override;
	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
	virtual void onEvent(const sf::Event& event) override;

private:
	//sf::Window& mWindow;
	PolygonGen mPolygons;

	std::vector<std::unique_ptr<Visualization>> mVisualizers;
	size_t mVisualizerIndex = 0;

	DrawStack mDrawStack;
	int mSimulationIndex = 0;

};