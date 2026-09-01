---
layout: post
title: "[ML] FiLM Block"
date: 2026-08-11 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine Learning
---
## FiLM (Feature-wise Linear Modulation ) Block

A convolutional layer produces feature channels that represent learned concepts such as edges, shadows, textures, or object responses. FiLM uses the condition to decide, for each channel:

- how strongly that channel should matter;

- whether its activation should be shifted upward or downward.

Conceptually:

```text
FiLM: MLP → gamma, beta

image → Conv → FiLM → ReLU → Conv → residual output
```

### What FiLM actually does

Suppose the feature map has shape `[B, C, H, W]`, and the condition has shape `[B, D]`. The conditioning network produces `2C` values:

```text
z → [gamma, beta]
```

There is one `gamma` and one `beta` per feature channel. FiLM applies:

`modulated_features = (1 + gamma) * features + beta`

The values are broadcast over the spatial dimensions, so every pixel in a given channel receives the same channel-level modulation.

The distinction between FiLM and residual is:

```text
residual:
output = input + learned_correction

FiLM:
output = condition_dependent_scale * features
       + condition_dependent_shift
```

FiLM changes the representation that the following layers receive. A residual connection determines how the block combines its input with its learned correction. They solve different problems and can be used together without being redundant.

### Why use FiLM at all?

Without FiLM, the convolutional path must use the same feature-processing behavior for every condition. For example, the network may learn a generic “shadow” channel, but it must then use the same response regardless of:

- signal-to-noise ratio;
- range regime;
- sensor configuration;
- noise level;
- acquisition mode.

If we have pre-trained a feautre network, we can freeze that, and attach multiple classifier / regression heads with FiLM  blocks. Those FiLM blocks allow the condition to alter the interpretation of those channels.

For one condition, the network might produce:

```text
gamma = [0.5, 1.0, -0.8]
beta  = [0.0, 0.2, -0.1]
```

The corresponding scales are:

```text
1 + gamma = [1.5, 2.0, 0.2]
```

This means:

- strengthen the first feature channel;

- strongly emphasize the second;

- suppress the third.

The `beta` values shift the activations. When FiLM is followed by ReLU, those shifts can make a channel more or less likely to activate.

The important point is that the condition is not being added as another image-like signal. It is being used to control the behavior of the feature extractor.

### Is FiLM itself a residual?

No, although the `1 + gamma` parameterization has a residual-like identity behavior.

With:

`output = (1 + gamma) * features + beta`

zero modulation gives:

`output = features`

That makes FiLM safe to initialize as an identity transformation. However, FiLM is still a multiplicative and additive modulation, not an additive skip connection.

A residual block says:

```text
output = input + correction
```

FiLM says:

```text
output = modify(features according to condition)
```

The residual connection preserves information from the block input. FiLM controls the internal feature processing. They are complementary.

### FiLM inside a residual block

A typical conditional residual block is:

```python
def forward(self, x, condition):
    residual = x

    h = self.conv1(x)
    h = self.film(h, condition)
    h = F.relu(h)
    h = self.conv2(h)

    return residual + h
```

This can be understood as:

```text
output = x + condition-dependent_correction(x)
```

The correction is condition-dependent because FiLM changes the intermediate features before the second convolution.

The skip connection is useful because the block can preserve the original representation and learn only the condition-dependent change. FiLM is useful because it tells the correction path how to behave under the current condition.

There is no contradiction in using both:

- the residual connection controls information flow across the block;

- FiLM controls the behavior inside the block.

### Why use `1 + gamma`?

The original FiLM equation is often written as:

`output = gamma * features + beta`

Your implementation uses:

`output = (1 + gamma) * features + beta`

This makes zero modulation the identity:

`output = (1 + 0) * features + 0 = features`

You can initialize the modulation layer to zero:

```python
class FiLM(nn.Module):
    def __init__(self, feature_channels, condition_features):
        super().__init__()

        self.modulation = nn.Linear(
            condition_features,
            2 * feature_channels,
        )

        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features, condition):
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * features + beta
```

At the beginning of training, FiLM does nothing. The network first behaves like an ordinary convolutional or residual network, then gradually learns how the condition should modify the features.

### FiLM versus conditional batch normalization

FiLM applies:

`output = gamma(z) * features + beta(z)`

Conditional batch normalization applies the same kind of condition-dependent scale and shift after normalization:

`output = gamma(z) * normalized_features + beta(z)`

FiLM does not require batch normalization. It is simply a learned affine transformation controlled by the condition.

---

## [From Original Paper](https://arxiv.org/pdf/1709.07871): FiLM as Language-Conditioned Visual Computation

FiLM allows one input to control how a neural network processes another input. In the original visual-reasoning model, an RNN interprets a natural-language question while a CNN processes the image. FiLM provides the connection between them.

The architecture contains two pipelines:

```text
Question → word embeddings → GRU → question embedding
                                      │
                                      ▼
                         gamma and beta parameters

Image → CNN features → FiLM ResBlocks → pooling → classifier
```

The GRU reads the question one word at a time and produces a final question embedding. A learned projection converts that embedding into separate `gamma` and `beta` parameters for each FiLM residual block.

Suppose an intermediate CNN feature map has shape `[B, C, H, W]`. The FiLM generator produces one `gamma` and one `beta` for each of the `C` channels:

```text
gamma, beta = Linear(question_embedding)
```

The original paper applies:

```text
modulated_features = gamma * features + beta
```

The same `gamma` and `beta` are broadcast over every spatial position in a channel. FiLM therefore performs channel-wise rather than pixel-wise conditioning.

For example, the CNN may contain channels that respond to purple objects, cubes, spheres, or spatial relationships. Given the question:

> What shape is the purple thing?

FiLM can strengthen **channels** related to purple objects and object shape while suppressing irrelevant channels. For a different question, the same CNN processes the same image using different channel scales and shifts.

This is what the authors mean when they say that an RNN over the question influences CNN computation over the image. The RNN does not directly process the image or repeatedly communicate with the CNN. Its final question representation generates parameters that control the CNN’s intermediate features. The complete system is trained end to end using the final answer loss.

### What the normalization experiments established

The authors moved FiLM to different locations inside the residual blocks and measured CLEVR validation accuracy:

```text
Best architecture                    97.4 ± 0.4%
FiLM after the second ReLU           97.7%
FiLM after the second convolution    97.1%
FiLM after the residual connection   96.6%
No batch normalization               93.7%
```

FiLM remained effective when it was moved away from the normalization layer. Placing it after the second ReLU achieved 97.7%, which was within or slightly above the best architecture’s reported range.

This shows that FiLM’s conditioning ability does not depend on being attached directly to normalization. It does not show that normalization is useless. Removing batch normalization reduced accuracy to 93.7%, suggesting that normalization still helped training and final performance.

The precise conclusion is:

> FiLM conditioning does not require normalization, although normalization may still improve optimization, regularization, and accuracy.

### How channel-wise FiLM produces attention-like spatial localization

Standard FiLM is spatially uniform: every location in one channel receives the same scale and shift. Nevertheless, the paper shows that FiLM can indirectly produce spatially selective behavior.

This happens because CNN feature maps are already spatially varying:

1. The CNN produces feature responses at different image locations.

2. FiLM strengthens question-relevant channels and suppresses irrelevant ones.

3. ReLU determines which modulated activations continue through the network.

4. Later convolutions combine the selected features locally.

5. Repeated FiLM blocks progressively emphasize locations containing relevant combinations of attributes and relationships.

For example, channels representing `purple`, `object`, and `cube` will jointly respond most strongly where the purple cube appears. FiLM controls which feature types matter, while the convolutional network determines where those features occur.

Overall, the figure supports the paper’s central claim: question-dependent channel modulation can produce attention-like spatial behavior without an explicit spatial-attention module. The RNN determines which visual features matter, the CNN determines where those features occur, and the final classifier uses the resulting representation to answer the question.

---

## Dilated FiLM: Combining Wider Context with Conditional Control

“Dilated FiLM” is not a separate mathematical version of FiLM. It usually means a residual block that combines:

- **dilated convolutions**, which examine a larger spatial region, and controls how far the network can see
- **FiLM conditioning**, which determines which feature channels matter under the current condition.

A dilated FiLM residual block might look like this:

```text
                         condition z
                              │
                              ▼
                         gamma, beta
                              │
                              ▼
input → dilated Conv → FiLM → ReLU → Conv → add input
```

FiLM still applies its usual channel-wise transformation:

```text
modulated_features = (1 + gamma) * features + beta
```

Dilation does not change `gamma` or `beta`. It changes the feature map that FiLM acts on.

The dilated convolution may learn channels representing:

- broad shadows;

- extended edges;

- regional texture;

- long noise streaks;

- relationships between separated structures.

FiLM then uses the condition to strengthen or suppress those channels. For example:

```text
high SNR → emphasize fine local structure
low SNR  → emphasize broader contextual patterns
far range → emphasize large-scale noise or shadow features
```

The network learns these behaviors from data. They are not manually assigned to particular channels.

### Stacking dilation rates

A common design uses progressively larger dilation rates:

```text
Block 1: dilation = 1
Block 2: dilation = 2
Block 3: dilation = 4
Block 4: dilation = 8
```

For one `3 × 3` convolution per block, the theoretical receptive field is:

```text
R = 1 + 2 * (1 + 2 + 4 + 8)
  = 31
```

The final block can therefore depend on a `31 × 31` region without downsampling the feature maps. If every block contains two `3 × 3` convolutions at its assigned dilation, the receptive field becomes:

```text
R = 1 + 4 * (1 + 2 + 4 + 8)
  = 61
```

FiLM can generate different parameters for every block:

```text
condition → gamma_1, beta_1
          → gamma_2, beta_2
          → gamma_3, beta_3
          → gamma_4, beta_4
```

The condition can then influence both local features in the early blocks and broader contextual features in the later blocks.

```python
class DilatedFiLMBlock(nn.Module):
    def __init__(self, channels, condition_features, dilation):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.film = FiLM(
            feature_channels=channels,
            condition_features=condition_features,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, condition):
        residual = x

        h = self.conv1(x)
        h = self.film(h, condition)
        h = F.relu(h)
        h = self.conv2(h)

        return residual + h
```

Setting `padding=dilation` preserves the image height and width for a `3 × 3` dilated convolution.

### Limitation: gridding artifacts

Dilated convolutions sample sparse positions. Repeatedly using the same large dilation can divide the feature map into disconnected sampling grids. Fine structures between sampled locations may be missed.

A safer schedule mixes local and dilated convolutions:

```text
dilation = 1 → 2 → 4 → 1
```

Another option is to process the same feature map through parallel branches:

```text
             ┌→ dilation 1 ─┐
features ────┼→ dilation 2 ─┼→ combine → FiLM
             └→ dilation 4 ─┘
```

This captures several spatial scales without relying on one sparse grid.
