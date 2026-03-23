#include "parser.h"
#include <fstream>
#include <iostream>
#include <cstdio>

using namespace std;

static vector<int64_t> extract_shape(const onnx::ValueInfoProto& v) {
    vector<int64_t> shape;
    if (!v.has_type()) return shape;
    const auto& t = v.type();
    if (!t.has_tensor_type()) return shape;
    const auto& tt = t.tensor_type();
    if (!tt.has_shape()) return shape;
    for (const auto& dim : tt.shape().dim()) {
        if (dim.has_dim_value())
            shape.push_back(dim.dim_value());
        else
            shape.push_back(-1);  // dynamic dim
    }
    return shape;
}

static void print_shape(const vector<string>& names) {
    cout << "[";
    for (size_t i = 0; i < names.size(); i++) {
        if (i) cout << ", ";
        cout << names[i];
    }
    cout << "]";
}

Parser::Parser(const string& path) {
    // Store directory so we can find external data files later.
    size_t slash = path.rfind('/');
    model_dir_ = (slash != string::npos) ? path.substr(0, slash + 1) : "./";

    ifstream input(path, ios::in | ios::binary);
    if (!input) {
        cerr << "error: cannot open " << path << "\n";
        exit(1);
    }
    if (!model_.ParseFromIstream(&input)) {
        cerr << "error: failed to parse model\n";
        exit(1);
    }
}

void Parser::parse() {
    const onnx::GraphProto& graph = model_.graph();

    // --- Build the shape map ---
    // Shape information is scattered across four sources in the ONNX graph.
    // We collect all of them into a single map keyed by tensor name,
    // so that when printing nodes we can look up any input/output shape in O(log n).

    // Weights and biases: read shape + float data in one pass.
    map<string, ifstream> open_files;
    size_t total_bytes = 0;

    for (const auto& t : graph.initializer()) {
        TensorInfo tensor;
        tensor.name  = t.name();
        tensor.shape = {t.dims().begin(), t.dims().end()};

        // How many floats?
        size_t count = 1;
        for (int64_t d : tensor.shape) count *= d;

        // Read the actual float data.
        if (t.data_location() == onnx::TensorProto::EXTERNAL) {
            string location;
            size_t offset = 0, length = 0;
            for (int i = 0; i < t.external_data_size(); i++) {
                if (t.external_data(i).key() == "location")
                    location = t.external_data(i).value();
                else if (t.external_data(i).key() == "offset")
                    offset = stoull(t.external_data(i).value());
                else if (t.external_data(i).key() == "length")
                    length = stoull(t.external_data(i).value());
            }
            string filepath = model_dir_ + location;
            if (open_files.find(filepath) == open_files.end()) {
                open_files[filepath] = ifstream(filepath, ios::binary);
                if (!open_files[filepath]) {
                    cerr << "error: cannot open weight file " << filepath << "\n";
                    exit(1);
                }
            }
            ifstream& file = open_files[filepath];
            tensor.data.resize(count);
            file.seekg(offset);
            file.read(reinterpret_cast<char*>(tensor.data.data()), length);
        } else if (t.raw_data().size() > 0) {
            if (t.data_type() == onnx::TensorProto::INT64) {
                // Convert int64 raw data to float.
                tensor.data.resize(count);
                const int64_t* src = reinterpret_cast<const int64_t*>(t.raw_data().data());
                for (size_t j = 0; j < count; j++)
                    tensor.data[j] = static_cast<float>(src[j]);
            } else {
                tensor.data.resize(count);
                memcpy(tensor.data.data(), t.raw_data().data(), count * sizeof(float));
            }
        } else if (t.float_data_size() > 0) {
            tensor.data.assign(t.float_data().begin(), t.float_data().end());
        } else if (t.int64_data_size() > 0) {
            tensor.data.resize(count);
            for (size_t j = 0; j < count; j++)
                tensor.data[j] = static_cast<float>(t.int64_data(j));
        }

        total_bytes += tensor.data.size() * sizeof(float);
        g.shapes[t.name()] = move(tensor);
    }

    cout << "Loaded " << graph.initializer_size() << " weights ("
         << total_bytes / 1024 / 1024 << "."
         << (total_bytes / 1024 * 10 / 1024) % 10 << " MB)\n";

    // Model input (e.g. the image tensor [1, 3, 224, 224]).
    // Only add if not already present (don't overwrite initializer data).
    for (const auto& v : graph.input()) {
        if (g.shapes.find(v.name()) == g.shapes.end()) {
            TensorInfo tensor;
            tensor.name  = v.name();
            tensor.shape = extract_shape(v);
            g.shapes[v.name()] = tensor;
        }
    }

    // Intermediate activation tensors produced between nodes.
    for (const auto& v : graph.value_info()) {
        if (g.shapes.find(v.name()) == g.shapes.end()) {
            TensorInfo tensor;
            tensor.name  = v.name();
            tensor.shape = extract_shape(v);
            g.shapes[v.name()] = tensor;
        }
    }

    // Model output (e.g. class logits [1, 1000]).
    for (const auto& v : graph.output()) {
        if (g.shapes.find(v.name()) == g.shapes.end()) {
            TensorInfo tensor;
            tensor.name  = v.name();
            tensor.shape = extract_shape(v);
            g.shapes[v.name()] = tensor;
        }
    }

    // --- Build the node list ---
    for (int i = 0; i < graph.node_size(); i++) {
        const auto& n = graph.node(i);
        Node node;
        node.op_type = n.op_type();
        node.inputs  = {n.input().begin(),  n.input().end()};
        node.outputs = {n.output().begin(), n.output().end()};
        for (const auto& attr : n.attribute()) {
            if (attr.ints_size() > 0)
                node.attrs[attr.name()] = {attr.ints().begin(), attr.ints().end()};
            else if (attr.has_i())
                node.attrs[attr.name()] = {attr.i()};
        }
        g.nodes.push_back(node);
    }
}

void Parser::print() const {
    cout << "Graph: " << g.nodes.size() << " nodes, " << g.shapes.size() << " shapes\n";
    for (size_t i = 0; i < g.nodes.size(); i++) {
        cout << "[" << i << "] " << g.nodes[i].op_type << " in: ";
        print_shape(g.nodes[i].inputs);
        cout << " out: ";
        print_shape(g.nodes[i].outputs);
        cout << endl;
    }
}

// Helper: write a JSON string, escaping special characters.
static void json_string(ofstream& out, const string& s) {
    out << '"';
    for (char c : s) {
        if (c == '"') out << "\\\"";
        else if (c == '\\') out << "\\\\";
        else out << c;
    }
    out << '"';
}

void Parser::export_json(const string& path) const {
    ofstream out(path);
    if (!out) {
        cerr << "error: cannot write " << path << "\n";
        return;
    }

    out << "{\n";

    // --- Nodes ---
    out << "  \"nodes\": [\n";
    for (size_t i = 0; i < g.nodes.size(); i++) {
        const Node& n = g.nodes[i];
        out << "    {\n";
        out << "      \"id\": " << i << ",\n";
        out << "      \"op\": "; json_string(out, n.op_type); out << ",\n";

        out << "      \"inputs\": [";
        for (size_t j = 0; j < n.inputs.size(); j++) {
            if (j) out << ", ";
            json_string(out, n.inputs[j]);
        }
        out << "],\n";

        out << "      \"outputs\": [";
        for (size_t j = 0; j < n.outputs.size(); j++) {
            if (j) out << ", ";
            json_string(out, n.outputs[j]);
        }
        out << "],\n";

        out << "      \"attrs\": {";
        bool first_attr = true;
        for (const auto& [key, vals] : n.attrs) {
            if (!first_attr) out << ", ";
            first_attr = false;
            json_string(out, key);
            out << ": [";
            for (size_t k = 0; k < vals.size(); k++) {
                if (k) out << ", ";
                out << vals[k];
            }
            out << "]";
        }
        out << "}\n";

        out << "    }" << (i + 1 < g.nodes.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    // --- Shapes (tensor metadata) ---
    // Include byte offset and length for tensors with data so the web
    // engine can load weights from a companion binary file.
    out << "  \"tensors\": {\n";
    size_t si = 0;
    size_t weight_offset = 0;
    for (const auto& [name, info] : g.shapes) {
        out << "    "; json_string(out, name); out << ": {\n";
        out << "      \"shape\": [";
        for (size_t j = 0; j < info.shape.size(); j++) {
            if (j) out << ", ";
            out << info.shape[j];
        }
        out << "],\n";
        bool has = !info.data.empty();
        out << "      \"has_data\": " << (has ? "true" : "false");
        if (has) {
            size_t byte_len = info.data.size() * sizeof(float);
            out << ",\n      \"offset\": " << weight_offset;
            out << ",\n      \"byte_length\": " << byte_len;
            weight_offset += byte_len;
        }
        out << "\n";
        out << "    }" << (si + 1 < g.shapes.size() ? "," : "") << "\n";
        si++;
    }
    out << "  }\n";

    out << "}\n";
    out.close();

    cout << "Exported graph to " << path << "\n";
}

void Parser::export_weights(const string& path) {
    ofstream out(path, ios::binary);
    if (!out) {
        cerr << "error: cannot write " << path << "\n";
        return;
    }

    size_t total = 0;
    // Iterate in the same order as export_json (std::map = sorted by key)
    for (const auto& [name, info] : g.shapes) {
        if (!info.data.empty()) {
            out.write(reinterpret_cast<const char*>(info.data.data()),
                      info.data.size() * sizeof(float));
            total += info.data.size() * sizeof(float);
        }
    }
    out.close();

    cout << "Exported weights to " << path
         << " (" << total / 1024 / 1024 << "."
         << (total / 1024 * 10 / 1024) % 10 << " MB)\n";
}