#pragma once

#include "octreebuilder_api.h"

#include <vector>
#include <unordered_map>
#include <iosfwd>

#include "octantid.h"

namespace octreebuilder {

/**
 * @brief The LinearOctree class stores a list of octants that are inside the tree's bounds.
 *
 * Octants are identified by thier morton code and level (OctantID).
 * Usually only the leafs are stored. However this is not guaranteed by the data structure.
 */
class OCTREEBUILDER_API LinearOctree {
public:
    typedef ::std::vector<OctantID> container_type;

    /**
     * @brief Creates a minimal empty linear octree (can't contain any leafs)
     */
    LinearOctree();

    /**
     * @brief Creates an linear octree
     * @param root The root of the octree
     * @param leafs The leafs of the octree
     */
    LinearOctree(const OctantID& root, const container_type& leafs = {});

    /**
     * @brief Creates an empty linear octree
     * @param root The root of the octree
     * @param numLeafs The expected number of leafs (preallocates space for numLeafs)
     */
    LinearOctree(const OctantID& root, const size_t& numLeafs);

    /**
     * @brief The root of the tree.
     *
     * Defines the bounds of the tree.
     * Hence all stored octants are decendants of the root.
     */
    const OctantID& root() const;

    /**
     * @brief The depth of the tree (distance from root node to level 0).
     */
    uint depth() const;

    /**
     * @brief The id's of the octants stored in the tree.
     */
    const container_type& leafs() const;

    /**
     * @brief adds The octant to the tree (as the last item).
     * @param octant The id of the octant.
     *
     * Its not allowed to add duplicates (however its not checked in release mode).
     */
    void insert(const OctantID& octant);
    void insert(container_type::const_iterator begin, container_type::const_iterator end);

    /**
     * @brief Checks if the octant is stored in the linear tree.
     * @param octant The id of the octant.
     * @return true If octant is stored inside the tree and false otherwise.
     * @pre sortAndRemove Must have been called before without calling a non const method in between.
     */
    bool hasLeaf(const OctantID& octant) const;

    /**
     * @brief Replaces a octant with its 8 children.
     * @param octant The octant to replace.
     * @return The ids of the children.
     * @note The octant is only marked for removal and is still returned by the leafs method. To finally erase it sortAndRemove has to be called.
     *
     * Ignored if called twice for the same octant without calling sortAndRemove in between.
     * The octant to replace doesn't have to exist in the list, however its more efficient
     * to use the insert method in that case.
     */
    ::std::vector<OctantID> replaceWithChildren(const OctantID& octant);

    /**
     * @brief Replaces an octant with the given subtree.
     * @param octant The id of the octant to replace.
     * @param subtree The octants of the subtree.
     * @note The octant is only marked for removal and is still returned by the leafs method. To finally erase it sortAndRemove has to be called.
     *
     * Each OctantID in the subtree should be a descendant of octant.
     * However this is not checked.
     * Ignored if called twice for the same octant without calling sortAndRemove in between.
     * The octant to replace doesn't have to exist in the list, however its more efficient
     * to use the insert method in that case.
     */
    void replaceWithSubtree(const OctantID& octant, const ::std::vector<OctantID>& subtree);

    /**
     * @brief For an octant A find the octant B in the linear tree, so that A > B and B is maximal (there is no other C in the tree with A > C and C > B).
     * @param octant The octant for which the lower bounds will be found.
     * @param lowerBound On success will refer to the lower bound of octant when the function returns. Otherwise the reference will be unchanged.
     * @return true If the search was successfull, false otherwise (there is no lower bound).
     * @pre sortAndRemove Must have been called before without calling a non const method in between.
     */
    bool maximumLowerBound(const OctantID& octant, OctantID& lowerBound) const;

    /**
     * @brief Checks wether the octant is inside the bounds of the tree.
     * @param octant The id of the octant.
     * @return true If the octant is inside the tree bounds, false otherwise.
     */
    bool insideTreeBounds(const OctantID& octant) const;

    /**
     * @brief The octant with the maximal id (level zero and maximal morton code) that is inside the tree bounds.
     * @note Does not depend on the octants stored octants.
     */
    OctantID deepestLastDecendant() const;

    /**
     * @brief The octant of level zero with the minimal id that is inside the tree bounds.
     * @note Does not depend on the octants stored octants.
     */
    OctantID deepestFirstDecendant() const;

    /**
     * @brief Sorts the stored octants in ascending order by their id and erases all octants that where marked for removal.
     */
    void sortAndRemove();

    /**
     * @brief Attempt to preallocate enough memory for specified number of leafs.
     */
    void reserve(const size_t numLeafs);

private:
    OctantID m_root;
    OctantID m_deepestLastDecendant;
    container_type m_leafs;
    ::std::unordered_map<morton_t, uint> m_toRemove;
};

OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const LinearOctree& octree);
}
