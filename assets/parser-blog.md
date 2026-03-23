# What is this website?

How does a machine look at a cat photo and know it's a cat? (Or fail to — because yes, they fail sometimes.) That question is exactly why I built this. I want to show you how a neural network actually works underneath, no hand-waving, no magic. I'm going to explain it from scratch using a real model called **MobileNetV2**, made by Google to identify images.

**Why MobileNetV2?** Because it's tiny. Most powerful models are gigabytes and need serious hardware. MobileNetV2 was designed to run on phones, so it's small enough to run in a browser — which is exactly what we're doing here. It's not the most accurate thing in the world, but it does a solid job and it's perfect for learning.

This first page covers the basics and parsing. The next page — hit **Inference** in the top right — covers actually running the model.

Let's dive in.

## What is a neural network?

You can find a hundred explanations online. I'll give you the one that worked for me — not too deep, but not shallow either.

Think of brain neurons. Lots of small nodes connected to each other, sending signals back and forth. A neural network is the same idea, except we built it, and instead of biological signals it does **math**. That's it. Nodes connected in order, doing math, producing a result.

Since MobileNetV2 is built for images, it takes an image as input, turns it into numbers, passes those numbers through all its nodes doing math, and spits out a result at the end. That result is a probability — *is this 98% cat? 15% dog? 0.05% plane?* It's probably a cat.

So how do we turn an image into numbers?

### Images are just pixels

Every image is made of pixels — tiny squares arranged in a grid. Take a 7×7 image: 49 pixels, 7 rows, 7 columns. But pixels aren't all the same. Each one has a color, and every color can be described mathematically using **RGB** (red, green, blue). Mix those three in different amounts and you get any color. Your monitor works this exact way.

We represent each channel on a scale from **0 to 255**. So a pixel isn't one number — it's three. One for red, one for green, one for blue.

That means our 7×7 image is actually **7 × 7 × 3 = 147 numbers**.

> This block of numbers — a multi-dimensional array — is called a **tensor**. Fancy word, simple idea.

### What the computer actually sees

Take that weird shape from above. Each pixel becomes three numbers: black pixels → `[0, 0, 0]`, coral red → `[230, 90, 75]`, gold → `[230, 195, 90]`, cyan → `[115, 210, 195]`. The result is three separate 7×7 grids stacked on top of each other — one per color channel. No shapes, no colors. Just numbers.

MobileNetV2 expects images at **224×224**, so instead of 147 numbers you're feeding it:

```
224 × 224 × 3 = 150,528 numbers
```

All from one image. That's why neural networks need to be fast.

### Layers

Those nodes we mentioned are organized into **layers**:

| Layer | Role |
|-------|------|
| **Input layer** | Receives the image tensor |
| **Hidden layers** | Do the heavy math, one after another |
| **Output layer** | Produces the final answer |

Each layer takes the output of the previous one, does its own math on it, and passes the result forward. Data flows in one direction — left to right — until an answer comes out.

## What's actually happening inside?

The main operation in every layer is **matrix multiplication**. Sounds scary. It isn't.

Say a layer has 3 numbers coming in. The layer also has its own set of numbers called **weights**. To compute the output, you multiply each input by its matching weight and add everything up:

```
input:   [2,   3,   1  ]
weights: [0.5, 0.8, 0.2]

(2 × 0.5) + (3 × 0.8) + (1 × 0.2) = 1.0 + 2.4 + 0.2 = 3.6
```

One output number. Now imagine doing this hundreds of thousands of times with larger tensors. That's MobileNetV2. Each layer multiplies the incoming tensor by its weights, passes the result to the next layer, repeat.

### The output

The final layer produces **1000 numbers** — one score per category (cat, dog, plane, banana, you name it). Raw scores are hard to read, so we pass them through **softmax**, which converts them into percentages that add up to 100%:

> *"91% cat, 6% fox, 1% dog"* → The model says: cat.

### Where do the weights come from?

Weights are the numbers the model *learned* during training. Someone fed it millions of labeled images and the model slowly adjusted its weights, step by step, until it got good at telling things apart. Think of it like a baby learning — starts with random guesses, gets corrected, improves over time. Millions of iterations.

We don't train it. We just use the weights that are already learned.

So now you know three things:

1. **Input** → a tensor (image as numbers)
2. **Math** → matrix multiplication (inputs × weights, summed up)
3. **Weights** → learned numbers, stored in a file

That file is what we need to parse.

## What is parsing and why do we need it?

When someone trains a model like MobileNetV2, everything gets saved to a file — the layers, the order they run, and all the weights. In our case that's an **`.onnx` file**. ONNX stands for *Open Neural Network Exchange*, a standard format for saving and sharing trained models across different tools.

Think of it like a ZIP file for a neural network. It has everything. It's a manual.

The problem: it's written in binary. You can't open it in a text editor and read it. Parsing is the act of **reading that file and making sense of it** — pulling out the structure (what layers exist, in what order) and the weights (the numbers each layer uses to do its math). Without parsing, it's just a blob of bytes that means nothing.

> It's like receiving a manual written in a language you don't speak. Before you can follow it, you have to translate it.

And here's the part I'm proud of: **I wrote the parser myself**. No library that does it all for us. We read the ONNX file directly, pull out the layers and weights, and build the whole map from scratch.

## The parser

So how do we actually read a binary ONNX file?

When you download the `.onnx` file, you also get a **`.proto` file** — short for Protocol Buffers, a format made by Google. The ONNX file is packed in a specific binary layout and the proto file is the spec that tells you how to unpack it: *"the first bytes are the model name, then here are the layers, then here are the weights."*

> The proto file is the dictionary. The ONNX file is the book written in that language. Use the dictionary to read the book.

Once parsed, you get a structured object containing:

- **What layers the model has** — convolution, batch norm, relu, etc. (you can see them as colored nodes in the graph)
- **What order they run in**
- **The input/output size of each layer** — e.g. `[1, 3, 224, 224]` for the image input
- **The weight values for each layer**

We loop through every layer, read its type, grab its weights, and store everything in our own data structure. That's the parser: read, understand, map.

## The graph you're looking at

We already ran the parser. The graph on the left is the result — all 100 nodes of MobileNetV2, in order. Click any node to inspect its inputs, outputs, weights, and attributes.

The colors tell you what kind of operation each node is:

| Color | Operation | What it does |
|-------|-----------|--------------|
| 🔵 Blue | **Convolution** | Slides a small window across the image looking for patterns — edges, corners, curves. Early layers catch simple things like lines. Later ones detect eyes, wheels, faces. |
| 🟢 Green | **Activation** | A signal gate. Decides which values to keep and which to zero out. Without this, the whole network collapses to one giant linear equation and fails at anything complex. |
| 🔴 Red | **Residual (Add)** | Adds the layer's input directly to its output. A clever bypass — even if a layer makes a mistake, the original signal still passes through. Think of it as a backup lane on a highway. |
| 🟣 Purple | **Gemm** | The final classifier — one big matrix multiply to turn the learned features into 1000 category scores. |

There are more types — hover or click to explore them.

### The order matters

You might wonder: *can I run all the Conv layers first, then the activations?* No. The order in the graph is the only correct order. Each node's output is the next node's input. You follow the chain from left to right, node by node, until you reach the end and get your answer. The researchers who trained the model determined this order — it's baked in.

We now have a human-readable map of MobileNetV2. The structure is clear, the weights are loaded, and every operation is accounted for.

**Hit Inference →** to see the model actually run.
