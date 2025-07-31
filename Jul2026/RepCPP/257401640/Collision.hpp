#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <span>

using Hull = std::span<glm::vec2>; // TODO eh?
using Object = std::span<glm::vec2>;
using CollisionID = size_t;
using Pairs = std::vector<std::pair<CollisionID, CollisionID>>;

class NarrowPhaseDetector
{
public:
	explicit NarrowPhaseDetector(const Hull& a, const Hull& b) 
		: mHullA(a)
		, mHullB(b)
	{} 

	virtual operator bool() = 0; // this probably killed performance

protected:
	Hull mHullA;
	Hull mHullB;
};

class BroadPhaseDetector
{
public:

	void addCollider(Object object)
	{
		mObjects.emplace_back(object);
		onColliderAddition();
	}

	std::vector<Object>& getObjects() { return mObjects; }
	Object& getObject(size_t i) { return mObjects[i]; }

	virtual Pairs generatePairs() = 0;

protected:
	virtual void onColliderAddition() = 0;

protected:
	std::vector<Object> mObjects;

};

class SpatialGrid : public BroadPhaseDetector
{
public:
	SpatialGrid(size_t gridSize);

	virtual Pairs generatePairs() override;

private:
	virtual void onColliderAddition() override;
	void makeBins();

private:
	std::vector<std::vector<CollisionID>> mBins;
	glm::uvec2 mGridSpan;
	size_t mGridSize;
};

class CollisionDetector
{
public:
	void addCollider(Object object);
	void setBroadPhaseDetector(std::unique_ptr<BroadPhaseDetector>&& detector);

	void update();
	std::vector<CollisionID> queryCollision(CollisionID id);
	bool queryIsColliding(CollisionID id);

private:
	std::unique_ptr<BroadPhaseDetector> mBroadphase;
	Pairs mCollisions;
};