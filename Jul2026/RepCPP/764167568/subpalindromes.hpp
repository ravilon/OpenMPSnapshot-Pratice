#ifndef SUBPALINDROMES_HPP
#define SUBPALINDROMES_HPP

#include <string>
#include <utility>
#include <vector>
#include <cstdint>

namespace ALGO
{

struct subpali_info_pos
{
  std::size_t odd;
  std::size_t even;
};

using subpali_info = std::vector<subpali_info_pos>;

template <typename CharT>
subpali_info find_subpalindromes_trivial(const std::basic_string<CharT>& source)
{
  auto len = source.size();
  subpali_info results(len);

#pragma omp parallel default(none) shared(results, len, source)
  #pragma omp for
  for (std::size_t idx = 0U; idx != len; ++idx)
  {
    results[idx].odd  = 1U;
    results[idx].even = 0U;

    // Odd-length subpalindromes
    while (idx >= results[idx].odd && idx + results[idx].odd < len
           && source[idx - results[idx].odd] == source[idx + results[idx].odd])
    {
      ++results[idx].odd;
    }

    // Even-length subpalindromes
    while (idx >= results[idx].even + 1U && idx + results[idx].even < len
           && source[idx - results[idx].even - 1U] == source[idx + results[idx].even])
    {
      ++results[idx].even;
    }
  }

  return results;
}

template <typename CharT>
subpali_info find_subpalindromes_manaker(const std::basic_string<CharT>& source)
{
  subpali_info results(source.size());
  auto num = static_cast<intmax_t>(source.size());

  // Odd-length subpalindromes
  {
    // Left and right borders of the rightmost subpalindrome
    intmax_t l = 0, r = -1;

    for (intmax_t idx = 0; idx < num; ++idx)
    {
      intmax_t k =
        (idx > r) ? 1 : std::min(static_cast<intmax_t>(results[l + r - idx].odd), r - idx + 1);

      while (idx + k < num && idx >= k && source[idx + k] == source[idx - k])
      {
        ++k;
      }

      results[idx].odd = k;
      if (idx + k - 1 > r)
      {
        // Update left and right borders
        l = idx - k + 1;
        r = idx + k - 1;
      }
    }
  }

  // Even-length subpalindromes
  {
    // Left and right borders of the rightmost subpalindrome
    intmax_t l = 0, r = -1;

    for (intmax_t idx = 0; idx < num; ++idx)
    {
      intmax_t k =
        (idx > r) ? 0 : std::min(static_cast<intmax_t>(results[l + r - idx + 1].even), r - idx + 1);

      while (idx + k < num && idx >= k + 1 && source[idx + k] == source[idx - k - 1])
      {
        ++k;
      }

      results[idx].even = k;
      if (idx + k - 1 > r)
      {
        // Update left and right borders
        l = idx - k;
        r = idx + k - 1;
      }
    }
  }

  return results;
}

} // namespace ALGO

#endif /* SUBPALINDROMES_HPP */
