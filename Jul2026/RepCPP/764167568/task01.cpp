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

using row = double[Jsize];
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

/* Main computational cycle. */
#pragma omp parallel default(none) shared(a)
#pragma omp for schedule(static)
  for (int i = 0; i < Isize; ++i)
  {
    for (int j = 0; j < Jsize; ++j)
    {
      a[i][j] = sin(2 * a[i][j]);
    }
  }

#ifdef TIMING
  std::clog << "Total elapsed: " << sw.stop() << " sec \n";
#endif

#ifndef QUIET
  auto result_file = std::ofstream("result.txt");
  assert(result_file.is_open());

  for (int i = 0; i < Isize; ++i)
  {
    for (int j = 0; j < Jsize; ++j)
    {
      result_file << a[i][j] << ' ';
    }

    result_file << std::endl;
  }
#endif /* !QUIET */

  delete[] a;
}
