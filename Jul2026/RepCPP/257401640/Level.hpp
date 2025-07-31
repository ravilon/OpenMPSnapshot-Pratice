#pragma once
#include <SFML/Graphics.hpp>

class Level : public sf::Drawable, public sf::Transformable
{
public:
	virtual void update(float dt) = 0;
	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const = 0;
	virtual void onEvent(const sf::Event& event) = 0;
};