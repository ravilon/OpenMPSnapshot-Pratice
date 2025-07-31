#pragma once

#include <string>
#include <iterator>
#include <fstream>

class InputIterator
{
private:
  std::ifstream *input_stream;
  std::string current_line;

public:
  InputIterator();
  InputIterator(std::ifstream *input_stream);
  ~InputIterator();
  std::string operator*();
  InputIterator &operator++();
  bool operator!=(const InputIterator &other);
};

class Input
{
private:
  std::ifstream input_stream;

public:
  Input(const char *sourcefile);
  ~Input();
  InputIterator begin();
  InputIterator end();
  std::string operator*();
};