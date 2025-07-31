#pragma once
#include <glm/glm.hpp>
#include <SFML/Graphics.hpp>

#include "Level.hpp"
#include "PolygonGen.hpp"
#include "Collision.hpp"


class PerfBench : public Level
{
	static constexpr float MOVESPEED = 0.5f;

public:
	PerfBench(sf::Window& window);

	virtual void update(float dt) override;
	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
	virtual void onEvent(const sf::Event& event) override;

private:
	void updatePolygons(float dt);

private:
	sf::Window& mWindow;
	PolygonGen mPolygons;
	CollisionDetector mColliDetector;

	std::vector<sf::ConvexShape> mShapes;
};