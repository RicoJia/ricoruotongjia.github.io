---
layout: post
title: Math - Stats Basics Recap
subtitle: Basic Statistics Concepts, Regression, Distributions, Covariance & Correlation, Bessel Correction, box filter
date: 2017-02-04 13:19
header-img: img/bg-material.jpg
tags:
  - Math
---

## Basic Statistics Concepts

### Standard Deviation and Variance

Variance of a distribution can be "biased" and "unbiased". A biased variance is to always underestimate the real bias.

$$
\begin{gather*}
\text{unbiased variance} = \frac{\sum (x - \bar{x})}{n-1}
\\
\text{biased variance} = \frac{\sum (x - \bar{x})}{n}

\end{gather*}
$$

The above unbiasing operation is caleld "Bessel correction"

An Isotropic covariance matrix is:

$$
\begin{gather*}
\begin{aligned}
& C = \lambda I
\end{aligned}
\end{gather*}
$$

### Stochastic Processes and Stationarity

A stochastic process is a collection of Random Variables over time. If a random variable is `X`, a process of it is `X(t)`. If the mean and variance of `X(t)` does not change, then loosely, this process is stationary.

#### Reasoning For Bessel Correction

- Population variance (or the true variance of the entire population) is calculated as:

$$
\begin{gather*}
\sigma^2 = \frac{1}{N} \sum_N (x - \mu)^2
\end{gather*}
$$
    - Where `N` is the whole popilation's size, $\mu$ is the population mean

- Sample variance:

$$
\begin{gather*}
s^2 = \frac{1}{n} \sum_n (x - \bar{x})^2
\end{gather*}
$$
    - Where `n` is the batch size, $\bar{x}$ is the batch mean

The sample variance has a slight bias because $\bar{x}$ is a random variable dependent on the sample. The population mean is slightly larger, so we divide by $N-1$ instead of $N$.

## Distributions

### Student's t-distribution

Student's t-distribution is similar to a Gaussian distribution, but with heavier tails and shorter peaks.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e1b05729-6a45-4f89-bb39-3ecf4b3cb047" height="300" alt=""/>
    </figure>
</p>
</div>

TODO

## Covariance And Correlation

Given two random variables $A$, $B$

- Mean

$$
\begin{gather*}
\mu_A = \frac{1}{N} \sum_i A_i, \mu_B = \frac{1}{N} \sum_i B_i
\end{gather*}
$$

- Standard Deviation

$$
\begin{gather*}
\sigma_A = \sqrt{\frac{1}{N} \sum_i (A_i - \mu_A)^2},
\sigma_B = \sqrt{\frac{1}{N} \sum_i (B_i - \mu_B)^2}
\end{gather*}
$$

    - Standard Deviation indicates how spread out the data is from the mean. 

- Covariance

$$
\begin{gather*}
cov(AB) = \frac{1}{N} \sum_i (A_i - \mu_A)(B_i - \mu_B)
\end{gather*}
$$

    - Covariance indicates **the Joint variability of $(A_i, B_i)$ pairs together.** If a single pair of $A_i$, $B_i$ are both positive, you will get a positive value. If one of them is positive, one of them is negative, you will get a negative value. Altogether, they could indicate how related $A$ and $B$ are. 

- Correlation

$$
\begin{gather*}
corr(AB) = \frac{cov(AB)}{\sigma_A \sigma_B}
\end{gather*}
$$
    - Correlation is a standardized measure of "relatedness" between two random variables. It ranges from $[-1, 1]$. If $A=kB$ after mean normalization, then correlation will be a perfect 1

### Covariance of Function

For a simple nonlinear function

$$  
y=f(x),  
$$

linearized around (x_0),

$$  
y \approx f(x_0)+f'(x_0)\delta x.  
$$

If (x) has covariance (P_x), then the covariance of (y) is approximately

$$  
\boxed{  
P_y=f'(x_0)P_xf'(x_0)^\top.  
}  
$$

## Regression

Regression in statistics means "estimating the relationship model between dependent variables and independent variables." For example, linear regression models the relationship of `Y` and its independent variables, `x1, x2 ...`.

```
Y=β0​+β1​x1​+β2​x2​+⋯+βn​xn​+ϵ
```

Transformer is "autoregressive". It's regressive because it tries to model the relationship between output and input sequences. It's "auto" because the output sequence depends on the previous output.

## Random Process

A random process $R(t)$ is basically a collection of random variables that vary along time. The random variables' mean and standard deviations may or may not change. If they don't change, we call the random process **stationary**

### Gaussian Random Process

A Gaussain Random Process is

$$
\begin{gather*}
\begin{aligned}
& R(t) \sim \mathcal{gp}(m(t), k(t, t') )
\end{aligned}
\end{gather*}
$$

Where the mean function of the Random Process is $m(t)$, and the **covariance function** $k(t, t')$ could change over time, too.

$$
\begin{gather*}
\begin{aligned}
& m(t) = E[R(t)]

\\
& k(t, t') = E[(R(t) - m(t))(R(t') - m(t'))]
\end{aligned}
\end{gather*}
$$

One special case is white Gaussian Random noise:

$$
\begin{gather*}
\begin{aligned}
& R(t) \sim \mathcal{gp}(0, \delta(t - t') \sigma^2)
\end{aligned}
\end{gather*}
$$

the covariance $\sigma$ does not change across time. Between different times, `t, t'`, there's no correlation between them, and they are independent.
$\delta(t - t')$ is "Dirac Delta Distribution."  It's a probability distribution, where everywhere is 0 except for at time `t'`. Also, $\int_{-\infty}^{\infty} \delta(t-t')f(t) = f(t')$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e6b0a7b9-8ee2-4f6b-a85f-78c1534110aa" height="300" alt=""/>
    </figure>
</p>
</div>

### Power Spectral Density

In signal procesisng, if we view a signal `x(t)` as a random process, then we can find its power across all frequencies. This is called "power spectral density" (PSD).

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/aec3b75d-2846-4e33-b57a-c4930049a028" height="300" alt=""/>
        <figcaption><a href="https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html">Source: Mathworks</a></figcaption>
    </figure>
</p>
</div>

It's defined as the Fourier Transform of the auto-correlation of the signal function at time difference $\tau$. The autocorrelation is:

$$
\begin{gather*}
\begin{aligned}
& R_{xx}(\tau) = E[x(t)x(t + \tau)]
\end{aligned}
\end{gather*}
$$

So if the signal is periodic with period of $\tau$, $R_{xx}(n\tau)$ would peak. The PSD $S_{xx}(f)$ is then the Fourier Transform of the autocorrelation across all time differences, $\tau$:

$$
\begin{gather*}
\begin{aligned}
& S_{xx}(f) = F( R_{xx}(\tau)) = \int_{-\infty}^{\infty} R_{xx}(\tau) e^{-j2\pi f \tau} d\tau
\end{aligned}
\end{gather*}
$$

For white Gaussian noise, the PSD is a **constant** $\sigma^2$:

$$
\begin{gather*}
\begin{aligned}
& S_{xx}(f) = F( R_{xx}(\tau)) = \int_{-\infty}^{\infty} \sigma^2 \delta(\tau) e^{-j2\pi f \tau} d\tau = \sigma^2
\end{aligned}
\end{gather*}
$$

### Wiener Process

Wiener Process is a.k.a Brownian Motion. It's non-stationary

$$
\begin{gather*}
\begin{aligned}
& W(t + \Delta t) = W(t) + \Delta W

\\
& \Delta W \sim \mathcal{N}(0, \Delta t)
\end{aligned}
\end{gather*}
$$

Its mean is 0, but variance is t (so it increases)

$$
\begin{gather*}
\begin{aligned}
& E[W(t)] = 0
\\ & Var[W(t)] = t
\end{aligned}
\end{gather*}
$$

## Box Filter

a box filter is a same-size local average, one output value per input pixel, not a downsample
