---
layout: post
title: "[ML] Neural Architecture Search (NAS) in RF DETR"
date: 2026-08-14 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine Learning
---
The main idea of the NAS RF-DETR is different resolutions do produce different size tensors and therefore different sized computation graphs, but they can still use the same learned weights. Training varies resolution, patch size, and window count, but during the NAS training described in the appendix, RF-DETR still uses 6 decoder layers and 300 queries on every training iteration. Decoder layers and queries are dropped later during architecture search / inference.

Imagine our RF-DETR has these learned parameters:

$$  
\theta =  
{  
W_{\text{patch}},  
W_{\text{ViT}},  
W_{D1},  
W_{D2},  
\dots,  
W_{D6},  
W_{\text{head}}  
}.  
$$
Now suppose training allows two resolutions: $320\times320$ or $640\times640$. For simplicity, suppose patch size is fixed at $p=16$.  Suppose the original image is large - RF-DETR chooses $r=320$ for the batch and resizes the images. The input tensor is:

$$  
X  
\in  
\mathbb{R}^{B\times3\times320\times320}.  
$$
Now divide it into `16x16` patches: along one dimension, each patch gets
$$  
320/16=20.  
$$
Per side, then each patch gets $20\times20=400$ image tokens. Then, 400 tokens -> ViT -> 300 queries -> decoder layers

$$  

X_{320}  
\rightarrow  
\boxed{W_{\text{patch}}}  
\rightarrow  
400\text{ tokens}  
\rightarrow  
\boxed{W_{\text{ViT}}}  
\rightarrow  
300\text{ queries}  
\rightarrow  
D_1  
\rightarrow  
D_2  
\rightarrow  
D_3  
\rightarrow  
D_4  
\rightarrow  
D_5  
\rightarrow  
D_6.  
$$

Then through back propagation, all these weight layers get trained.

Now let's say NAS samples $r=640$.  The new batch is

$$  
X  
\in  
\mathbb{R}^{B\times3\times640\times640}.  
$$With the same patch size,

$$  
640/16=40.  
$$

Therefore we now have

$$  
40\times40=1600  
$$

tokens. Then, ViT produces 400 encoder features, one per spatial token: $Z\in\mathbb{R}^{B\times400\times d}$ . Then, we feed them into a proposal head evaluates each feature and predicts:

- An object/class confidence
- An initial bounding box

we choose the top 300 features by their scores. Similarly, for the larger image, we choose 300 top features from 1600 proposals. Then the rest of the network uses the same weights.
