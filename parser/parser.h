#pragma once
#include "onnx.pb.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

// Holds tensor metadata and optionally the actual float data (for weights).
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;  // populated for weights, empty for activations
};

// Represents a single compute node: the operation type, its input/output tensor names,
// and integer attributes (e.g. strides, pads, group for Conv).
struct Node {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::vector<int64_t>> attrs;  // attribute name -> int list
};

// The parsed graph: ordered list of nodes and a name-to-shape lookup for all tensors.
struct Graph {
    std::vector<Node> nodes;
    std::map<std::string, TensorInfo> shapes;  // tensor name -> shape metadata
};

class Parser {
public:
    Graph g;

    explicit Parser(const std::string& path);
    void parse();
    void print() const;
    void export_json(const std::string& path) const;
    void export_weights(const std::string& path);

    // Provides read access to the parsed ONNX model proto.
    const onnx::ModelProto& model() const { return model_; }

private:
    onnx::ModelProto model_;
    std::string model_dir_;  // directory containing the .onnx file
};
