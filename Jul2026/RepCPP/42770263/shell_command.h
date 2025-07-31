#pragma once
#include "typedefs.h"
#include <vector>
#include <string>
#include <iostream>

class ShellCommand { //all shell commands will be inherited from this class.
protected:
    std::string name;
public:
  ShellCommand();

//starts a command with provided args, which include command name.
  virtual void run(std::string arguments);

//returns the name of a command
  std::string getName() {
    return name;
  }

  //checks whether the number of arguments is correct, used in all commands.
  bool checkNumberOfArguments(int real, int expected, std::ostream& out);

  //does the same thing as previous command, but with two possible variants.
  bool checkNumberOfArguments(int real, int expected1, int expected2, std::ostream& out);

  //parses command arguments, used in all commands.
  std::vector<std::string> parseArguments(std::string notParsedArguments);

};
