---
layout: post
title: Deep Learning - Transformer Series 3 - Multi-Head and Self Attention
date: '2022-03-28 13:19'
subtitle: Multi-Head Attention, Self Attention, Comparison of Self Attention Against CNN, RNN
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Multi-Head Attention

To learn a richer set of behaviors, we can instantiate multiple attentions jointly given the **same set of queries, keys, and values**. Specifically, we are able to capture various long-range and short range dependencies.

The Process is:

1. Linearly transform `q`, `k`, `v` into `q'`, `k'`, `v'`. We have made sure they all have the same hidden dimension `hidden_size`**
    - This adds learnability for the non-linear decision landscape.
1. Split `q'`, `k'`, `v'` into heads: `h1_q`, `h1_k`, `h1_v`, `h2_q`, `h2_k`, `h2_v`. A head is a part of the overall `q'`, `k'`, `v'`.
1. The attention module is [additive or scaled-product attention pooling](./2022-03-27-deep-learning-attention-mechanism.markdown). The attention module does not have any learnable parameters. **They run on each head in parallel**.
    - For each head $i$, attention is calculated based on its unique $W_i^Q Q$, $W_i^K K$ , $W_i^V V$
    - In the "Attention is All You Need" paper, 8 heads were used.
1. All `h` are concatenated
3. The concatenated head is transformed into a shorter embedding through a dense layer, `Wo`

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/152c1a01-8123-449c-8f50-09b1cc6c967a" height="600" alt=""/>
       </figure>
    </p>
</div>

- I'm omitting the lenght masking part in the illustration. In reality we add it to focus on the generally relavant segment of the input sentence.

The reason why multi-headed attention works so well is:

- **Each input word will evolve into embeddings (i.e., key and value). Then the embeddings are divided into heads, where each head could represent a different meaning of the word. So, attention weights are given to different meanings of each individual word, based on the sub-vectors of other words.**
- Finally, the overall weighed-attention sub-vectors are calculated, concatenated together, transformed into an overall embedding.

One might notice that the linear transformations and the final attention pooling action (just the dot-product part) share the same weights across heads. This keeps the model small, yet still appears to be effective in real life.

- In the code below, the embedding is of length `hidden_size`

$$
\begin{gather*}
W_o [h_1, ... h_n]
\end{gather*}
$$

Now, let's enjoy the code. [The PyTorch Implementation is here, in case it's useful](https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L3854).

**[The below implementation has been tested against the PyTorch Implementation](https://github.com/RicoJia/Machine_Learning/blob/ffda794938c913b54a5316d1dca6d553393f0328/RicoModels_pkg/ricomodels/tests/test_og_transformer.py)**

```python
"""
Notes:
- Lazy*: borrow the nice input channel inference feature from TensorFlow
    e.g., torch.nn.LazyLinear(out_features=hidden_size, bias=False)
"""
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        1. Linearly transform q, k, v so that they all have the same hidden dimension hidden_size
        2. Split q', k', v' into heads
        3. Each group of q, k, v go into DotProductAttention
        4. The concatenated head is transformed into a shorter embedding through a dense layer, Wo
        """
        # embed_dim is also qk_dim,
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"Embed_dim: {embed_dim} must be divisible by num_heads: {num_heads}"
        # Doing Wq, Wk, Wv. By default, v is also assumed to be of length embed_dim
        self.Wq = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        # self.Wo
        self.out_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=False
        )  # TODO: by default, o is also of embed_dim?
        self.attention = DotProductAttention()
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.embedding_dim = embed_dim

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        """
        Args: ACHTUNG: THIS IS WEIRD because num_queries is at the front
        q (torch.Tensor): [num_queries, batch_size, qk_dim]
        k (torch.Tensor): [num_keys, batch_size, qk_dim]
        v (torch.Tensor): [num_keys, batch_size, v_dim]
        """
        num_queries, batch_size, _ = q.size()
        num_keys = k.size(0)
        q_proj = self.Wq(q)  # [num_queries, batch_size, embed_dim]
        k_proj = self.Wk(k)  # [num_keys, batch_size, embed_dim]
        v_proj = self.Wv(v)  # [num_keys, batch_size, embed_dim]
        # now, split them into num_heads. How to calculate heads in parallel?
        q = q_proj.view(num_queries, batch_size, self.num_heads, self.head_dim)
        k = k_proj.view(num_keys, batch_size, self.num_heads, self.head_dim)
        v = v_proj.view(num_keys, batch_size, self.num_heads, self.head_dim)

        # [batch, head_num, num_keys/num_queries, embed_dim]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        # [batch_size, head_num, query_num, head_embed_dim]
        attention = self.attention(
            q=q, k=k, v=v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        # [query_num, batch_size, head_num, head_embed_dim]
        attention = attention.permute(2, 0, 1, 3).contiguous()  # TODO? .contiguous()
        attention = attention.view(num_queries, batch_size, self.embedding_dim)
        attention_output = self.out_proj(attention)
        return attention_output
```

#### VERY IMPORTANT NOTE ABOUT `key_padding_mask` and `attn_mask` (or lookahead_mask)

[In the PyTorch implementation, attention_weight is set to `-inf` at `1` in `key_padding_mask`](https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L4119C31-L4119C50). This is right before `softmax()`, so the intetion is to have zero after `softmax` at these locations. However, in reality, we get could get `NaN`. The real reason is that `attn_mask` could mask out the rest of the `attention_weight`

- [An issue was opened in 2019 about this.](https://github.com/pytorch/pytorch/issues/24816). When a full vector is `-inf`, there are definitely `NaN`
- So a good strategy is to:
  - Expand `key_padding_mask = [1, 1, query_num, kv_num]`, and `attn_mask = [batch_size, 1, 1, num_keys]`
  - Do a logical or and check for the all masking-out situation

## Self Attention

when key, value, and query come from the same set of inputs, they are called "self-attention" [1]. We **also want to make sure the output has the same dimension as the inputs**. Since value and queries are the same, this is equivalent to having `num_queries` input words, and having `num_queries` output words

Now let's illustrate with some code

```python
num_hiddens, num_heads = 100, 5
attention = MultiheadedAttention(hidden_size=num_hiddens, output_size=num_hiddens, num_heads=num_heads)
attention.eval()
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
output = attention(X, X, X) #(batch_size, num_queries, num_hiddens)
print(attention)
```

### Comparing CNN, RNN, and Self-Attention

Saywe are given an `n` input tokens. They are a `nxd` vector. We are outputting a sequence of `dxn` as well. We compare:

- Time complexity
- Sequential Operations: number of actions which takes place in sequence. They are bottlenecks of parallel computations
- Maximum Path Lengths: the length (or number of layers) needed to allow for any input element to be considered in any output sequence.
  - For example, with 1 layer CNN, the first input element is considered within the first output element, but not the subsequent ones as the kernel moves forward. We need to have more layers so that the first element is considered.
  - A shorter path between any combination of sequence positions makes learning long-range dependencies easier

#### For CNN

- Input and output channels are `d`; kernel size is `k`
- Time complexity: $O(nd^2k)$ because we need to go over all elements in the input and output filters
- Sequential operations: we need to calculate layer by layer, but we know that beforehand, so $O(1)$
- Maximum Path length: (receptive field size?) is $O(n/k)$, For example, x1, x5 are within the receptive fields of CNN

#### For RNN

- Say we have 1 layer, since we are outputting with the same dimension, the hidden state dimension is `d` as well.
- Time Complexity: weight matrices are `dxd`. In total, $O(nd^2)$
- Sequential Complexity: $O(n)$
- Maximum path length: $O(n)$ as we need to finish the entire $n$ timesteps so the last output sequence can technically see the first input element.

#### For Self Attention

- Time Complexity: weight matrices are `nxd`. In total, $O(n^2d)$
- Sequential Complexity: $O(1)$: we need to do linear transform, concatenate, and dense layer.
- Maximum path length: $O(1)$ as the single operation is able to consider all input elements.

**So, both CNN and self attention has a low number of sequential operations and are highly parallelizable. However, self attention will suffer from higher complexity when input sequence is long.**

## References

[1] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems (pp. 5998–6008).](https://arxiv.org/pdf/1706.03762)
