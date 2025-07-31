#include "QuadTree.hpp"
#include "Constants.hpp"

namespace
{
    size_t mMaxNodeObjects;
    size_t mMaxDepth;
}

QuadTreeDetector::QuadTreeDetector(size_t maxNodeObjects, size_t maxDepth)
{
    mMaxDepth = maxDepth;
    mMaxNodeObjects = maxNodeObjects;
}

Pairs QuadTreeDetector::generatePairs()
{
    mRoot = std::make_unique<QuadTreeNode>(0, -AREA_SIZE, AREA_SIZE);

    for (CollisionID i = 0; i < mObjects.size(); i++)
    {
        mTreeObjects[i].update();
        mRoot->insert(mTreeObjects[i]);
    }

    Pairs pairs;
    mRoot->findAllCollidingPairs(pairs);
    return pairs;
}

void QuadTreeDetector::onColliderAddition()
{
    mTreeObjects.push_back(QuadTreeObject(mObjects.back(), mObjects.size() - 1));
}

QuadTreeDetector::QuadTreeObject::QuadTreeObject(Object object, CollisionID objectID) :
    objectID(objectID),
    object(object)
{
}

void QuadTreeDetector::QuadTreeObject::update()
{
    minBound = glm::vec2(std::numeric_limits<float>::max());
    maxBound = glm::vec2(std::numeric_limits<float>::lowest());
    for (auto& vertex : object) // Compute AABB
    {
        minBound.x = glm::min(minBound.x, vertex.x);
        minBound.y = glm::min(minBound.y, vertex.y);
        maxBound.x = glm::max(maxBound.x, vertex.x);
        maxBound.y = glm::max(maxBound.y, vertex.y);
    }
}

QuadTreeDetector::QuadTreeNode::QuadTreeNode(size_t depth, glm::vec2 topLeft, glm::vec2 botRight) :
    mDepth(depth),
    mTopLeft(topLeft),
    mBotRight(botRight),
    mCenter((topLeft + botRight) * 0.5f)
{
}

bool QuadTreeDetector::QuadTreeNode::inBounds(QuadTreeObject& object)
{
    return object.minBound.x >= mTopLeft.x && object.minBound.y >= mTopLeft.y && object.maxBound.x <= mBotRight.x && object.maxBound.y <= mBotRight.y;
}

void QuadTreeDetector::QuadTreeNode::insert(QuadTreeObject& object)
{
    if (!inBounds(object))
        return;

    if (isLeaf())
    {
        if (mDepth >= mMaxDepth || mQuadObjects.size() < mMaxNodeObjects)
            mQuadObjects.push_back(&object);
        else
        {
            split();
            insert(object);
        }
    }
    else
    {
        auto i = getQuadrant(object);
        if (i < 4)
            mSubTrees[i]->insert(object);
        else
            mQuadObjects.push_back(&object);
    }
}

void QuadTreeDetector::QuadTreeNode::split()
{
    mSubTrees[0] = std::make_unique<QuadTreeNode>(mDepth + 1, mTopLeft, mCenter);
    mSubTrees[1] = std::make_unique<QuadTreeNode>(mDepth + 1, glm::vec2(mCenter.x, mTopLeft.y), glm::vec2(mBotRight.x, mCenter.y));
    mSubTrees[2] = std::make_unique<QuadTreeNode>(mDepth + 1, glm::vec2(mTopLeft.x, mCenter.y), glm::vec2(mCenter.x, mBotRight.y));
    mSubTrees[3] = std::make_unique<QuadTreeNode>(mDepth + 1, mCenter, mBotRight);

    auto newValues = std::vector<QuadTreeObject*>();

    for (size_t i = 0; i < mQuadObjects.size(); i++)
    {
        auto j = getQuadrant(*mQuadObjects[i]);
        if (j < 4)
            mSubTrees[j]->mQuadObjects.push_back(mQuadObjects[i]);
        else
            newValues.push_back(mQuadObjects[i]);
    }

    mQuadObjects = std::move(newValues);
}

bool QuadTreeDetector::QuadTreeNode::isLeaf()
{
    return !static_cast<bool>(mSubTrees[0]);
}

size_t QuadTreeDetector::QuadTreeNode::getQuadrant(QuadTreeObject& object)
{
    if (object.maxBound.x < mCenter.x)
    {
        if (object.maxBound.y < mCenter.y)
            return 0; // entirely top left
        else if (object.minBound.y >= mCenter.y)
            return 2; // entirely bot left
    }

    else if (object.minBound.x >= mCenter.x)
    {
        if (object.maxBound.y < mCenter.y)
            return 1; // entirely top right
        else if (object.minBound.y >= mCenter.y)
            return 3; // entirely bot right
    }
    
    return 4; // not contained entirely in any quadrant
}

void QuadTreeDetector::QuadTreeNode::findAllCollidingPairs(Pairs& pairs)
{
    for (size_t i = 0; i < mQuadObjects.size(); i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            if (mQuadObjects[i]->intersects(*mQuadObjects[j]))
            {
                #pragma omp critical (pairs)
                pairs.emplace_back(mQuadObjects[i]->objectID, mQuadObjects[j]->objectID);
            }
        }
    }
    if (!isLeaf())
    {
        #pragma omp parallel for
        for (size_t i = 0; i < mSubTrees.size(); i++)
        {
            for (size_t j = 0; j < mQuadObjects.size(); j++)
                mSubTrees[i]->findAllCollidingDescendants(*mQuadObjects[j], pairs);
        }
        for (auto& subTree : mSubTrees)
            subTree->findAllCollidingPairs(pairs);
    }
}

void QuadTreeDetector::QuadTreeNode::findAllCollidingDescendants(QuadTreeObject& object, Pairs& pairs)
{
    for (size_t i = 0; i < mQuadObjects.size(); i++)
    {
        if (object.intersects(*mQuadObjects[i]))
        {
            #pragma omp critical (pairs)
            pairs.emplace_back(object.objectID, mQuadObjects[i]->objectID);
        }
    }

    if (!isLeaf())
        for (size_t i = 0; i < mSubTrees.size(); i++)
            mSubTrees[i]->findAllCollidingDescendants(object, pairs);
}

bool QuadTreeDetector::QuadTreeNode::intersects(QuadTreeObject& object)
{
    return object.minBound.x < mBotRight.x && object.maxBound.x > mTopLeft.x && object.minBound.y < mBotRight.y && object.maxBound.y > mTopLeft.y;
}

bool QuadTreeDetector::QuadTreeObject::intersects(QuadTreeObject& object)
{
    return minBound.x < object.maxBound.x && maxBound.x > object.minBound.x && minBound.y < object.maxBound.y && maxBound.y > object.minBound.y;
}
