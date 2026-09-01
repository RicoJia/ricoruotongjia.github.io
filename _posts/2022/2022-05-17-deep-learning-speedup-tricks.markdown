---
layout: post
title: Deep Learning - Speedup Tricks
date: '2022-05-17 13:19'
subtitle: Torch Optimizer Tricks, Mixed Precision Training
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## General Speed-Up Tricks

- Data loading tricks:
 	- compared to `npz` ,`npy` is faster to load during training because it uses contiguous disk memory and support `mmap`
 	- use json for boxes with labels.
  		- JSONL means **JSON Lines**. Each line is one independent JSON object
- If you look to use albumentations for augmentation, sticking to the `[batch, H, W, Channels]` (channel last) could make data loading faster

- `tensor.contiguous()` creates a new tensor that uses contiguous blocks of memory. You might need this after `permute()`, `view()`, `transpose()`, where the underlying memory is not contiguous.

- `torch.cuda.empty_cache()` empties cached variables on GPU. So please do this **before sending anything to the GPU**

- `@torch.inference_mode()` turns off autograd overhead and can be used for evaluate. There's no gradient tensor whatsoever. Accordingly, tensors created under this mode are **immutable**. This is great for **model evaluation**

```python
@torch.inference_mode()
def evaluate_model(input_data):
    output = model(input_data)
    return output
```

    - `torch.no_grad():` might still have intermediate tensors that carry gradients, and **allows for operations that modify tensors in place.**

## Torch Optimizer Tricks

### `RMSProp`

`foreach` option updates weights in a vectorized, batched manner, based on gradients and moving averages like momentum. This is more efficient than python `for loops` and uses optimized linear algebra libraries, like `BLAS`, `cuBLAS`. Without `foreach`, the vanilla RMSProp would be:

```python
# each param is the weight tensor of a layer
for param in model.parameters():
    # Compute gradient g_i
    g_i = param.grad

    # Update running average of squared gradients
    s_i = alpha * s_i + (1 - alpha) * (g_i ** 2)

    # Update parameter
    param -= eta * g_i / (sqrt(s_i) + epsilon)
```

But with `foreach`, the GPU can calculate the stacked weights altogether with stacked gradients and running averages

```
[
  W^{(1)} (matrix of size m x n),
  b^{(1)} (vector of size n),
  W^{(2)} (matrix of size n x p),
  b^{(2)} (vector of size p)
]
```

Caveat:

- Slight numerical differences

### Cache Clearing And Memory Management

- Empty cache

```python
# Tensors are immediately cleared in GPU memory. By default it's not
torch.cuda.empty_cache()
# This resets the internal memory counter that tracks the peak memory usage on the GPU.
# After resetting, you can accurately track peak memory
torch.cuda.reset_max_memory_allocated()
# ensures that all preceding GPU operations have been completed before moving to the next operation.
torch.cuda.synchronize()
```

- Be cautious with operations like `.item()`, `.numpy()`, `.cpu()` as they can cause synchronization.**So move data to the CPU last**

    ```python
    # Not doing item() here because that's an implicit synchronization call
    # .cpu(), .numpy() have synchronization calls, too
    local_correct = (predicted_test == labels_test).sum()
    ```

  - `.sum()` is **not** calling the GPU
