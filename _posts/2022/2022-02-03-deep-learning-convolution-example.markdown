---
layout: post
title: Deep Learning - Convolution Example
date: 2022-02-01 13:19
subtitle: conv1d, conv2d, 1x1 convolution 
comments: true
header-img: img/home-bg-art.jpg
tags:
  - Deep Learning
---
**Equivariant** means that ==if you change an input by a specific rule or symmetry (like moving or rotating it), the output changes in a matching, predictable way==. Convolution is equivarent

## 1D Convolution Example

Unlike Digital Signal Processing, 1D convolution in deep learning does **not** flip the kernel — it is simply cross-correlation.

Input $x = [1, 2, 3, 4]$ (length $N=4$), kernel $h = [1, 0, -1]$ (length $K=3$), stride $S=1$.

**Output length:** $\lfloor (N - K) / S \rfloor + 1 = (4 - 3)/1 + 1 = 2$

**Output values:**

$$
y_1 = 1{\cdot}1 + 2{\cdot}0 + 3{\cdot}(-1) = -2
$$

$$
y_2 = 2{\cdot}1 + 3{\cdot}0 + 4{\cdot}(-1) = -2
$$

$$
y = [-2,\; -2]
$$

## 2D Convolution Example

Assume input is 4x4x2

```
# Channel 1
[  1   5   9  13  
   2   6  10  14  
   3   7  11  15  
   4   8  12  16 ]

# Channel 2
[ 1  1  1  1  
  2  2  2  2  
  3  3  3  3  
  4  4  4  4 ]
```

We want to do convolution with

```python
conv = nn.Conv2d(
    in_channels=2,
    out_channels=3,
    kernel_size=2,
    stride=1,
    padding=0,
    bias=False
)
```

internally, the kernel tensor is `weight shape = (out_channels = 3, in_channels = 2, kernel_size = 2, kernel_size = 2)`

The kernel tensor has shape `(out_channels=3, in_channels=2, kH=2, kW=2)`:

- **Output channel 1**

  | | Channel 1 Weights | Channel 2 Weights |
  |---|---|---|
  | Row 0 | `[1 0]` | `[1 1]` |
  | Row 1 | `[0 1]` | `[1 1]` |

- **Output channel 2**

  | | Channel 1 Weights | Channel 2 Weights |
  |---|---|---|
  | Row 0 | `[0 1]` | `[0 0]` |
  | Row 1 | `[1 0]` | `[0 1]` |

- **Output channel 3**

  | | Channel 1 Weights | Channel 2 Weights |
  |---|---|---|
  | Row 0 | `[ 1  1]` | `[-1 -1]` |
  | Row 1 | `[ 1  1]` | `[-1 -1]` |

To convolve (2×2 kernel, stride 1, no padding):

1. Slide the 2×2 kernel spatially over each input channel.
2. For each spatial location:
   1. Multiply elementwise with the input patch.
   2. Sum the 4 products → one scalar per input channel.
3. Sum the scalars across all input channels → one output value.

---

### Output channel 1

**Step 1 — convolve input channel 1 (A) with kernel `[[1,0],[0,1]]`**

Top-left patch example:

$$
\begin{bmatrix}1 & 5\\2 & 6\end{bmatrix} \odot \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}
= 1{\cdot}1 + 5{\cdot}0 + 2{\cdot}0 + 6{\cdot}1 = 7
$$

Full result (stride 1 across the 4×4 input → 3×3 output):

$$
A * K_{1}^{(1)} = \begin{bmatrix}7 & 15 & 23\\9 & 17 & 25\\11 & 19 & 27\end{bmatrix}
$$

**Step 2 — convolve input channel 2 (B) with kernel `[[1,1],[1,1]]`** (sum of patch):

$$
B * K_{1}^{(2)} = \begin{bmatrix}6 & 6 & 6\\10 & 10 & 10\\14 & 14 & 14\end{bmatrix}
$$

**Step 3 — sum over input channels:**

$$
Y_1 = (A * K_{1}^{(1)}) + (B * K_{1}^{(2)}) =
\begin{bmatrix}13 & 21 & 29\\19 & 27 & 35\\25 & 33 & 41\end{bmatrix}
$$

---

### Output channel 2

**A with `[[0,1],[1,0]]`** (anti-diagonal):

$$
A * K_{2}^{(1)} = \begin{bmatrix}7 & 15 & 23\\9 & 17 & 25\\11 & 19 & 27\end{bmatrix}
$$

**B with `[[0,0],[0,1]]`** (bottom-right only):

$$
B * K_{2}^{(2)} = \begin{bmatrix}2 & 2 & 2\\3 & 3 & 3\\4 & 4 & 4\end{bmatrix}
$$

$$
Y_2 = \begin{bmatrix}9 & 17 & 25\\12 & 20 & 28\\15 & 23 & 31\end{bmatrix}
$$

---

### Output channel 3

**A with `[[1,1],[1,1]]`** (sum of patch):

$$
A * K_{3}^{(1)} = \begin{bmatrix}14 & 30 & 46\\18 & 34 & 50\\22 & 38 & 54\end{bmatrix}
$$

**B with `[[-1,-1],[-1,-1]]`** (negative sum of patch):

$$
B * K_{3}^{(2)} = \begin{bmatrix}-6 & -6 & -6\\-10 & -10 & -10\\-14 & -14 & -14\end{bmatrix}
$$

$$
Y_3 = \begin{bmatrix}8 & 24 & 40\\8 & 24 & 40\\8 & 24 & 40\end{bmatrix}
$$

---

### Final output

Stacking $Y_1, Y_2, Y_3$ gives a tensor of shape **3×3×3** (height × width × out\_channels).

> **Shape formula:** spatial output = $\lfloor (N - K) / S \rfloor + 1 = (4 - 2)/1 + 1 = 3$, so output is $3 \times 3 \times 3$.

---

## 1x1 Convolution Example

A 1×1 convolution acts as a **per-pixel learned linear combination across channels** — no spatial mixing, just channel mixing at every spatial location independently.

Assume input shape `[C_in=2, H=2, W=2]`:

$$
A = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}, \quad
B = \begin{bmatrix}10 & 20\\30 & 40\end{bmatrix}
$$

Use a 1×1 convolution with `C_out=3`. Weight tensor shape: `(3, 2, 1, 1)`.

Now we define the kernel, where each per-output-channel weight vector $[w_1,\, w_2]$ is:

| Output channel | $w_1$ (ch 1) | $w_2$ (ch 2) |
|---|---|---|
| #1 | 1 | 0 |
| #2 | 0 | 1 |
| #3 | 1 | −1 |

Each output channel is $Y_k = w_1 \cdot A + w_2 \cdot B$:

$$
Y_1 = 1\cdot A + 0\cdot B = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}
$$

$$
Y_2 = 0\cdot A + 1\cdot B = \begin{bmatrix}10 & 20\\30 & 40\end{bmatrix}
$$

$$
Y_3 = 1\cdot A + (-1)\cdot B = \begin{bmatrix}1-10 & 2-20\\3-30 & 4-40\end{bmatrix} = \begin{bmatrix}-9 & -18\\-27 & -36\end{bmatrix}
$$

Final output shape: `[C_out=3, H=2, W=2]` — the spatial dimensions are unchanged.

```python
nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1)
```
