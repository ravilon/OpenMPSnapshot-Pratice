#include <cmath>

#ifdef TIMING
#include <iostream>
#include "stopwatch.hpp"
#endif

#ifndef QUIET
#include <fstream>
#include <cassert>
#endif

namespace
{
/* Array dimensions. */
const int Isize = 40000;
const int Jsize = 40000;

using row = double[Jsize + 3];
} /* anonymous namespace */

int main()
{
  auto a = new row[Isize];

  /* Preparation - fill array with some data. */
  for (int i = 0; i < Isize; ++i)
  {
    for (int j = 0; j < Jsize; ++j)
    {
      a[i][j] = 10 * i + j;
    }
  }

#ifdef TIMING
  UTILS::stopwatch sw;
  sw.start();
#endif

  /* Duplicate data for extra columns. */
  for (int i = 0; i < Isize; ++i)
  {
    for (int j = Jsize; j < Jsize + 3; ++j)
    {
      a[i][j] = a[i][j - 3];
    }
  }

  /* Main computations. */
  for (int i = 2; i < 4; ++i)
  {
    for (int j = 0; j < Jsize - 3; ++j)
    {
      a[i][j + 3] = 2 * a[i - 2][j + 3];
    }
  }

  for (int i = 4; i < Isize; ++i)
  {
    #pragma omp parallel default(none) shared(a, i)
    #pragma omp for
    for (int j = 0; j < Jsize - 3; ++j)
    {
      a[i][j + 3] = sin(5 * a[i - 2][j + 6]);
    }
  }

#ifdef TIMING
  std::clog << "Total elapsed: " << sw.stop() << " sec \n";
#endif

#ifndef QUIET
  auto result_file = std::ofstream("result.txt");
  assert(result_file.is_open());

  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < Jsize; ++j)
    {
      result_file << a[i][j] << ' ';
    }

    result_file << std::endl;
  }

  for (int i = 2; i < Isize; ++i)
  {
    for (int j = 3; j < Jsize + 3; ++j)
    {
      result_file << a[i][j] << ' ';
    }

    result_file << std::endl;
  }
#endif /* !QUIET */

  delete[] a;
}
