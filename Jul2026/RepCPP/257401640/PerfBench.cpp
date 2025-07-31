#include "PerfBench.hpp"
#include "Constants.hpp"
#include <SFML/Window/Keyboard.hpp>

#include "GJK.hpp"
#include "QuadTree.hpp"

using namespace glm;


PerfBench::PerfBench(sf::Window& window)
	: mWindow(window)
{
	size_t defaultPolygonCount = 1000;
	mPolygons.generatePolygons(defaultPolygonCount);

	// create shapes
	for (size_t i = 0; i < defaultPolygonCount; ++i)
	{
		const auto& polygon = mPolygons[i];
		
		sf::ConvexShape shape;
		shape.setPointCount(polygon.size());
		
		for (size_t j = 0; j < polygon.size(); ++j)
			shape.setPoint(j, reinterpret_cast<sf::Vector2f&>(polygon[j]));

		mShapes.emplace_back(shape);
	}

	//mColliDetector.setBroadPhaseDetector(std::make_unique<SpatialGrid>(225)); // TODO alg switching
	mColliDetector.setBroadPhaseDetector(std::make_unique<QuadTreeDetector>(10, 5));
	for (auto& p : mPolygons.data())
		mColliDetector.addCollider(p);
}

void PerfBench::update(float dt)
{
	// handle camera movement
	vec2 movedir = { 0, 0 };

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
		movedir.x += 1;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
		movedir.x -= 1;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
		movedir.y += 1;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
		movedir.y -= 1;

	movedir *= MOVESPEED;

	auto delta = movedir * dt;
	move(delta.x, delta.y);

	updatePolygons(dt);
	mColliDetector.update();

	auto position = getPosition();
	position.x = std::min(std::max(position.x, -AREA_SIZE.x + mWindow.getSize().x), AREA_SIZE.x);
	position.y = std::min(std::max(position.y, -AREA_SIZE.y + mWindow.getSize().y), AREA_SIZE.y);
	setPosition(position.x, position.y);

	for (size_t i = 0; i < mShapes.size(); ++i)
		mShapes[i].setFillColor(mColliDetector.queryIsColliding(i) ? sf::Color::Cyan : sf::Color::White);
}

void PerfBench::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
	// draw gridlines
	constexpr int stride = 225;

	std::vector<sf::Vertex> lines;
	for (size_t i = 0; i < mWindow.getSize().y / stride + 2; ++i)
	{
		float y = std::fmod(getPosition().y, stride) + i * stride;
		sf::Vertex a, b;

		a.position = { 0.f, y };
		b.position = { static_cast<float>(mWindow.getSize().x), y };
	
		lines.emplace_back(a);
		lines.emplace_back(b);
	}

	for (size_t i = 0; i < mWindow.getSize().x / stride + 2; ++i)
	{
		float x = std::fmod(getPosition().x, stride) + i * stride;
		sf::Vertex a, b;

		a.position = { x, 0.f };
		b.position = { x, static_cast<float>(mWindow.getSize().y) };

		lines.emplace_back(a);
		lines.emplace_back(b);
	}

	target.draw(lines.data(), lines.size(), sf::Lines, states);

	states.transform *= getTransform(); // getTransform() is defined by sf::Transformable
	for (auto& s : mShapes)
		target.draw(s, states);
}

void PerfBench::onEvent(const sf::Event& event)
{
}

void PerfBench::updatePolygons(float dt)
{
	//vec2 halfArea = { (MAX_AREA.x - mWindow.getSize().x) / 2, (MAX_AREA.y - mWindow.getSize().y) / 2 };
	
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < mPolygons.size(); ++i)
	{
		auto polygon = mPolygons[i];
		auto& direction = mPolygons.getDirections()[i];
		auto speed = direction * dt;
		
		bool pushX = false;
		bool pushY = false;
		for (auto& v : polygon)
		{
			v += speed;

			if (v.x < -AREA_SIZE.x || v.x > AREA_SIZE.x) 
				pushX = true;
			if (v.y < -AREA_SIZE.y || v.y > AREA_SIZE.y) 
				pushY = true;
		}

		if (pushX || pushY)
		{
			if (pushX) direction.x = -direction.x;
			if (pushY) direction.y = -direction.y;

			speed = direction * dt;
			for (auto& v : polygon)
				v += speed;
		}

		for (size_t j = 0; j < polygon.size(); ++j)
			mShapes[i].setPoint(j, reinterpret_cast<sf::Vector2f&>(polygon[j]));
	}
}
