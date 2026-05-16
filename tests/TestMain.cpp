#include "TestFramework.h"

#include <string>

int main(int argc, char **argv)
{
    std::string filter;
    if (argc > 1 && argv[1] != nullptr)
        filter = argv[1];

    return testfw::runAll(filter);
}
