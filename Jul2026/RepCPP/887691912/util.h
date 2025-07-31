#pragma once
#include <utility>
enum Direction {None, Top, Bottom, Left, Right};

struct Point {
  Point() : Point(0, 0) {}
  int x, y;
  Point(int x, int y) : x(x), y(y) {}
  bool operator==(const Point &p) const {
    return x == p.x && y == p.y;
  }
  bool operator!=(const Point &p) const {
    return !(*this == p);
  }
  bool operator<(const Point &p) const {
    return x < p.x || (x == p.x && y < p.y);
  }
};
