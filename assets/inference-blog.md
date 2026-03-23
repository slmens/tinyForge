# Now watch it think

The parser gave us a map. This page runs it.

You upload an image, hit Run, and the model walks through all 100 operations one by one — slow enough to see what's actually happening. Not a black box. The math, live, in front of you.

Let's trace exactly what happens from the moment you hit Run to the moment you see a result.

## Step 0 — Preprocessing

Before the model sees your image, we have to get it into the right shape. MobileNetV2 was trained on images that are **224×224 pixels**, so the first thing we do is resize yours to that exact size. Then we normalize the pixel values.

Raw pixels range from 0 to 255. The model expects values in a different range — roughly **−1 to +1** — because that's the range the training data was in. So every pixel gets transformed:

```
normalized = (pixel / 127.5) − 1
```

Black (0) becomes −1. White (255) becomes +1. Everything else lands somewhere in between. Now we have a tensor with shape `[1, 3, 224, 224]` — one image, three color channels, 224 rows, 224 columns. That's **150,528 numbers**. Hand it to the model.

## The 100 operations

MobileNetV2 has exactly 100 operations. They're not all the same — four types do most of the work. Here's what each one actually does.

### Convolution (52 nodes)

This is the workhorse. A convolution is a sliding window operation: take a small filter (usually 3×3 or 1×1 pixels), slide it across the entire image, and at each position compute a dot product between the filter and the patch of image underneath.

```
filter:    [0.1,  0.8, −0.3]
           [0.5, −0.2,  0.9]
           [0.0,  0.4,  0.2]

patch:     [120, 140,  90]   ← pixel values at this position
           [110, 200, 150]
           [ 80,  90, 160]

result:    (0.1×120) + (0.8×140) + (−0.3×90) + ...
```

Do that at every position across the whole image. You get an **output map** — a new grid of numbers. Run hundreds of different filters and you get hundreds of output maps. Each filter learns to detect something different: one might fire on horizontal edges, another on diagonal lines, another on curved shapes.

Early convolutions work on the full 224×224 image. By the end, the spatial resolution has shrunk to 7×7, but the depth has grown from 3 channels to 1280. The network is trading pixels for abstraction.

> You can't make a filter that detects "cat" directly. But stack enough filters and eventually one fires on whiskers, another on pointy ears, another on a nose — and the combination becomes "cat."

**Depthwise convolution** is a variant MobileNetV2 uses heavily. Instead of mixing all input channels together, it runs one filter per input channel independently. Much faster, same quality. That's the whole trick that makes it small enough to run in a browser.

### Clip / ReLU6 (35 nodes)

After a convolution, values can be anything — positive, negative, huge, tiny. Raw unbounded values make training unstable.

Clip forces them into a range: **clamp to [0, 6]**. Anything below 0 becomes 0. Anything above 6 becomes 6. Everything else stays as-is.

```
−3.7  →  0
 2.1  →  2.1
 8.4  →  6
 0.0  →  0
```

This is called **ReLU6** — it's like the classic ReLU (which just floors at zero) but with an upper cap too. The cap at 6 turns out to matter when the model gets compressed for phones. Keeps values in a predictable range.

Without this step, layers could amplify each other into enormous numbers that overwhelm the math further down the chain. The Clip nodes are the brakes.

### Add / Residual (10 nodes)

These are the **skip connections** — one of the most important ideas in modern deep learning.

Normally data flows one direction: layer → layer → layer. A skip connection creates a shortcut that adds the input of a block directly to its output.

```
block input tensor  +  block output tensor  =  new tensor
```

Why? Because sometimes the best thing a layer can do is *nothing* — just pass the input through unchanged. Without a skip, the layer has to learn that explicitly. With a skip, it can learn to output zero and the original signal passes through automatically.

More practically: they solve the vanishing gradient problem. During training, gradients travel backwards through the network. Without skips they fade out before reaching early layers, which then fail to learn. Skips give gradients a direct path all the way back. That's how you can train 100+ layer networks at all.

Look at them in the graph — the Add nodes always appear at the end of a block, reconnecting to where that block started.

### ReduceMean — Global Average Pooling (1 node)

At this point in the network, we have a tensor shaped `[1, 1280, 7, 7]` — 1280 feature maps, each 7×7 pixels.

ReduceMean collapses the spatial dimensions: for each of the 1280 channels, compute the average of all 49 values and keep just that one number.

```
1280 channels × 7 × 7 = 62,720 values
          ↓
1280 averages
```

One number per channel. No spatial info left. The 7×7 grid is gone, replaced by a single summary value: *on average, how much did this feature activate across the image?*

This is the point where the network stops thinking about **where** things are and starts thinking about **what** is there. The result is a vector of 1280 numbers — a compact signature of the whole image.

### Reshape (1 node)

Takes the tensor from `[1, 1280, 1, 1]` to `[1, 1280]`. The spatial dimensions are already 1×1 after ReduceMean, so this is just housekeeping — removing the extra dimensions so the final matrix multiply works cleanly.

### Gemm — Matrix Multiply (1 node)

The final operation. We have a **[1, 1280]** vector — the image's signature — and we multiply it by a **[1000, 1280]** weight matrix. The result is **[1, 1000]**: one score per class.

```
[1, 1280]  ×  [1000, 1280]ᵀ  =  [1, 1000]
```

This is the classifier. The 1000 rows of the weight matrix are 1000 learned "directions" in feature space — one pointing toward cat, one toward banana, one toward volcano. The dot product with each row measures how aligned the image's features are with that class.

You get 1000 raw scores called **logits**. They're not probabilities yet — just comparisons.

## The final answer

After all 100 operations, we have 1000 numbers. But raw logits are hard to interpret. **Softmax** converts them into proper probabilities:

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

Run every logit through `e^x`, then divide by the total. Now all 1000 values are between 0 and 1 and they add up to exactly 1. You have a probability distribution over 1000 classes.

The top result is the model's answer.

> "I'm 94% confident this is a tabby cat."

That's inference. No magic. Just 100 math steps, one image, one answer.

## What you're seeing

When you run the walkthrough, you're watching real computations. The code executes actual operations on your image's actual tensor data. The visualizations are samples — we can't draw all 62,720 values on one screen — but the math is real, step by step, the same math the original model runs.

The first convolution takes 150,528 numbers in. The last softmax puts out a probability over 1000 categories. Everything in between is the path between those two things.

That's it. That's the whole trick.
