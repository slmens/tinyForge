#include "parser/parser.h"
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: tinyforge <model.onnx> [--json output.json]\n";
        return 1;
    }

    Parser parser(argv[1]);
    parser.parse();
    parser.print();

    // Check for --json and --weights flags.
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            parser.export_json(argv[i + 1]);
        } else if (strcmp(argv[i], "--weights") == 0) {
            parser.export_weights(argv[i + 1]);
        }
    }

    return 0;
}
