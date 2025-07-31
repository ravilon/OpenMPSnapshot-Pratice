#pragma once

#include "Collision.hpp"
#include <set>
#include <array>

class QuadTreeDetector : public BroadPhaseDetector
{
public:
    QuadTreeDetector(size_t maxNodeObjects, size_t maxDepth);

    virtual Pairs generatePairs() override;
    virtual void onColliderAddition() override;

private:
    struct QuadTreeObject
    {
        glm::vec2 minBound;
        glm::vec2 maxBound;
        CollisionID objectID;
        Object object;
        QuadTreeObject(Object object, CollisionID objectID);
        void update();
        bool intersects(QuadTreeObject& object);
    };

    struct QuadTreeNode
    {
        QuadTreeNode(size_t depth, glm::vec2 topLeft, glm::vec2 botRight);
        bool inBounds(QuadTreeObject& object);
        void insert(QuadTreeObject& object);
        void split();
        bool isLeaf();
        size_t getQuadrant(QuadTreeObject& object);
        void findAllCollidingPairs(Pairs& pairs);
        void findAllCollidingDescendants(QuadTreeObject& object, Pairs& pairs);
        bool intersects(QuadTreeObject& object);

        size_t mDepth;
        glm::vec2 mTopLeft;
        glm::vec2 mBotRight;
        glm::vec2 mCenter;
        std::array<std::unique_ptr<QuadTreeNode>, 4> mSubTrees;
        std::vector<QuadTreeObject*> mQuadObjects;
    };

    std::unique_ptr<QuadTreeNode> mRoot;
    std::vector<QuadTreeObject> mTreeObjects;
};

