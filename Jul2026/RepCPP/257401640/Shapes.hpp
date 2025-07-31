#pragma once
#include <span>
#include <SFML/Graphics.hpp>
#include <glm/glm.hpp>


class ShapeHelper : public sf::Drawable, public sf::Transformable
{
public:
	ShapeHelper() {};
	virtual ~ShapeHelper() = default;
	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const
	{};
};

class Line : public ShapeHelper
{
public:
	explicit Line(const sf::Color color = sf::Color::White);
	virtual ~Line() = default;

	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
	void add(const glm::vec2& v);
	void add(const std::span<glm::vec2>& vertices, bool connect = false);
	void add(const std::vector<glm::vec2>& vertices, bool connect = false);

private:
	std::vector<sf::Vertex> mVertices;
	sf::Color mColor;
};

class Circle : public ShapeHelper
{
public:
	virtual ~Circle() = default;

	Circle(float radius = 0.f, sf::Color color = sf::Color::White) 
		: mShape(radius)
	{
		mShape.setOrigin(radius, radius);
		mShape.setFillColor(color);
	};

	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override
	{
		states.transform *= getTransform();
		target.draw(mShape, states);
	}

public:
	sf::CircleShape mShape;
};

class Text : public ShapeHelper
{
public:
	virtual ~Text() = default;

	Text(const std::string& str)
	{
		static sf::Font font;
		static bool loaded = false;
		if (!loaded)
		#if defined(_WIN32)
			font.loadFromFile("C:\\Windows\\Fonts\\Arial.ttf"), loaded = true;
		#elif defined(linux) || defined(__linux)
			font.loadFromFile("/usr/share/fonts"), loaded = true;
		#elif defined(_APPLE_) && defined(_MACH_)
			font.loadFromFile("/Library/Fonts"), loaded = true;
		#endif

		mText.setFillColor(sf::Color::White);
		mText.setCharacterSize(13);
		mText.setString(str);
		mText.setPosition(300, -250);
		mText.setFont(font);
	}

	virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override
	{
		states.transform *= getTransform();
		target.draw(mText, states);
	}

public:
	sf::Text mText;
};