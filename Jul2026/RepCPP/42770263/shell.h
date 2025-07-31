#pragma once
#include "shell_command.h"
#include <map>
#define PROMPT '$'

class Shell {//used for interactions with user, doesn't depend on program logics.
private:
  std::map <std::string, ShellCommand* > commands;//map from supported command name to command.
  std::ostream& out;//output stream.
  std::istream& in;//input stream.
public:
  Shell(const std::map<std::string, ShellCommand*>& newCommands, std::ostream& out, std::istream& in);
  //starts shell with commands, input stream and output stream provided  in constructor.
  void start ();
private:
  //parses command name from string.
  std::string getCommandName(const std::string& commandWithArgs) const;
  bool isSpaces(const std::string& str) const;
};
