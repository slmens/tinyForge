CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. -Iassets $(shell pkg-config --cflags protobuf)
LDFLAGS  = $(shell pkg-config --libs protobuf)
TARGET   = tinyforge
MODEL    = assets/mobilenetv2.onnx

SRCS_CPP = main.cpp parser/parser.cpp
SRCS_CC  = assets/onnx.pb.cc
OBJS     = $(SRCS_CPP:.cpp=.o) $(SRCS_CC:.cc=.o)

all: $(TARGET)
	@echo "Running: ./$(TARGET) $(MODEL)"
	@./$(TARGET) $(MODEL)

json: $(TARGET)
	@./$(TARGET) $(MODEL) --json assets/graph.json --weights assets/weights.bin

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

wasm: engine/inference.cpp
	emcc engine/inference.cpp -o web/inference.js \
		-s EXPORTED_FUNCTIONS='["_tf_alloc","_tf_free","_tf_conv","_tf_clip","_tf_add","_tf_reduce_mean","_tf_gemm","_tf_softmax","_malloc","_free"]' \
		-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32"]' \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s INITIAL_MEMORY=134217728 \
		-s MODULARIZE=1 \
		-s EXPORT_NAME='TinyForgeEngine' \
		-O2 \
		--no-entry
	@echo "Built web/inference.js + web/inference.wasm"

clean:
	rm -f $(OBJS) $(TARGET) web/inference.js web/inference.wasm

serve: json wasm
	@echo "Open http://localhost:8080/web/"
	@python3 -m http.server 8080

.PHONY: all clean json serve wasm
