#include <cstdio>
#include <string>
#include <queue>
#include <omp.h>


enum class Operation
{
  NONE,
  ADD,
  SUB,
  MUL,
  DIV
};

void calculate(const char* cmd, int& output)
{
  if (cmd == nullptr) {
    throw "Error: nullptr in calculate";
  }

  int result = 0;
  int value = 0;
  std::string raw = "";
  bool isProcessed = false;
  Operation op = Operation::ADD;

  for (int i = 0; !isProcessed; ++i) {
    switch (cmd[i]) {
      default: raw += cmd[i]; break;
      case '+': op = Operation::ADD; break;
      case '-': op = Operation::SUB; break;
      case '*': op = Operation::MUL; break;
      case '/': op = Operation::DIV; break;

      case 0:
        isProcessed = true;
      case ' ':
        if (raw.empty()) { 
          break;
        } else if (raw == "res") {
          value = output;
        } else {
          value = std::stoi(raw);
        }
        switch (op) {
          case Operation::ADD: result += value; break;
          case Operation::SUB: result -= value; break;
          case Operation::MUL: result *= value; break;
          case Operation::DIV: result /= value; break;
          default: throw "Error: undefined Operation";
        }
        op = Operation::NONE;
        raw.clear();
    }
  }

  output = result;
}


int main(int argc, char *argv[])
{
  const char* task[] = {
      "5 + 3",
      "res / 2",
      "24 / res",
      "res * res",
      "res - 6",
      "res / 10",
      nullptr
    };

  std::queue<const char*> pipeline;

  int result = 0;
  #pragma omp parallel sections
  {
    // BACKEND
    #pragma omp section
    {
      while (true) {
        if (pipeline.empty()) {
          continue;
        }

        const char* cmd = pipeline.front();
        if (cmd == nullptr) {
          break;
        }
        pipeline.pop();

        std::printf("Command recieved: %s\n", cmd);
        calculate(cmd, result);
        std::printf("Command executed: %s\n", cmd);
      }
    }

    // CLIENT
    #pragma omp section
    {
      for (auto cmd = task; *cmd != nullptr; ++cmd) {
        std::printf("Command pushed: %s\n", *cmd);
        pipeline.push(*cmd);
      }
      pipeline.push(nullptr);
    }
  }
  std::printf("Execution finished, res = %d\n", result);
} 