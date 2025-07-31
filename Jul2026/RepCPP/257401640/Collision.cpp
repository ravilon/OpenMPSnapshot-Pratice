#include "Collision.hpp"
#include "Constants.hpp"
#include "GJK.hpp"

#include <set>


using namespace glm;

SpatialGrid::SpatialGrid(size_t gridSize)
	: mGridSize(gridSize)
{
	mGridSpan = (static_cast<uvec2>(AREA_SIZE) * 2u) / static_cast<unsigned>(mGridSize) + 2u; // TODO check without +1
	mBins.resize(mGridSpan.x * mGridSpan.y);
}

Pairs SpatialGrid::generatePairs()
{
	makeBins();

	Pairs pairs; // TODO maybe reserve smth
	for (const auto& bin : mBins)
		for (size_t i = 0; i < bin.size(); ++i)
			for (size_t j = i + 1; j < bin.size(); ++j)
				pairs.emplace_back(bin[i], bin[j]);
	
	return pairs;
}

void SpatialGrid::makeBins()
{
	for (auto& b : mBins)
		b.clear();
	
	#pragma omp parallel for shared(mBins)
	for (CollisionID i = 0; i < mObjects.size(); ++i)
	{
		std::set<size_t> binSet;
		for (const auto& v : mObjects[i])
		{
			uvec2 gridpos = static_cast<uvec2>(v + AREA_SIZE + float(mGridSize / 2)) / static_cast<unsigned>(mGridSize);
			binSet.emplace(gridpos.x + gridpos.y * mGridSpan.x);
		}

		#pragma omp critical
		for (auto& b : binSet)
			mBins[b].emplace_back(i);
	}
}

void SpatialGrid::onColliderAddition()
{}

void CollisionDetector::addCollider(Object object)
{
	mBroadphase->addCollider(object);
}

void CollisionDetector::setBroadPhaseDetector(std::unique_ptr<BroadPhaseDetector>&& detector)
{
	mBroadphase.swap(detector);
}

void CollisionDetector::update()
{
	auto pairs = mBroadphase->generatePairs();
	mCollisions.clear();

	#pragma omp parallel for shared(mCollisions)
	for (size_t i = 0; i < pairs.size(); ++i)
	{
		const auto& p = pairs[i];
		if (GJK(mBroadphase->getObject(p.first), mBroadphase->getObject(p.second)))
		{
			#pragma omp critical
			mCollisions.emplace_back(p.first, p.second);
		}
	}
}

std::vector<CollisionID> CollisionDetector::queryCollision(CollisionID id)
{
	std::vector<CollisionID> query;

	for (const auto& c : mCollisions)
		if (c.first == id) query.emplace_back(c.second);
		else if (c.second == id) query.emplace_back(c.first);

	return query;
}

bool CollisionDetector::queryIsColliding(CollisionID id)
{
	for (const auto& c : mCollisions)
		if (c.first == id || c.second == id) return true;
	return false;
}
