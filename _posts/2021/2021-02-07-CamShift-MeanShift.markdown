---
layout: post
title: "Computer Vision - MeanShift and CamShift: Tracking an Object by Its Color"
date: 2021-02-07 13:19
subtitle:
comments: true
tags:
  - Computer Vision
---

MeanShift and CamShift are simple tracking algorithms. They do not understand that an object is a mine, vehicle, or person. Instead, they remember the object’s color distribution and search for pixels with similar colors in the next image.

The basic idea is:

1. Summarize the colors inside the original bounding box.

2. Give every pixel in the next image a color-matching weight.

3. Move the box toward the weighted center.

4. Measure how widely the matching pixels are distributed.

5. Resize and rotate the box to cover that distribution.

## Step 1: Build an HSV histogram

Suppose we draw a box around a yellow mine in the first image.

The image is converted from BGR to HSV:

- Hue describes the basic color, such as yellow or blue.

- Saturation describes how colorful the pixel is.

- Value describes how bright the pixel is.

In our example, the histogram uses hue and saturation. Value is only used to remove extremely dark pixels.

The hue range is divided into 30 bins:

$$
\text{hue-bin width} = \frac{180}{30} = 6
$$

A pixel’s hue-bin index is therefore:

$$
i = \left\lfloor \frac{H}{6} \right\rfloor
$$

The saturation range is divided into 32 bins:

$$
\text{saturation-bin width} = \frac{256}{32} = 8
$$

Its saturation-bin index is:

$$
j = \left\lfloor \frac{S}{8} \right\rfloor
$$

Here:

- H is the pixel’s hue.

- S is its saturation.

- i and j identify one bin in the two-dimensional histogram.

Every accepted pixel inside the original box increases the corresponding histogram count:

$$
C(i,j) \leftarrow C(i,j) + 1
$$

Suppose the box contains:

- 80 strongly saturated yellow pixels.

- 20 orange-yellow pixels.

The yellow bin gets a count of 80, while the orange-yellow bin gets a count of 20.

## Step 2: Convert histogram counts into weights

OpenCV normalizes the histogram to the range from 0 to 255.

When the smallest histogram count is zero, the normalized value is:

$$
\hat C(i,j) = 255 \frac{C(i,j)}{C_{\max}}
$$

For the yellow bin:

$$
\hat C_{\text{yellow}} = 255 \frac{80}{80} = 255
$$

For the orange-yellow bin:

$$
\hat C_{\text{orange}} = 255 \frac{20}{80} \approx 64
$$

Now consider a pixel at coordinates x and y in the second image.

OpenCV looks at that pixel’s hue and saturation, finds the corresponding histogram bin, and uses the bin value as the pixel’s weight:

$$
w(x,y) = \hat C \left( i(H(x,y)), j(S(x,y)) \right)
$$

Therefore:

- A pixel with the common yellow color receives a weight near 255.

- A pixel with the less-common orange-yellow color receives a weight near 64.

- A color not found in the original box receives a weight near zero.

- A pixel that is too dark or insufficiently saturated is forced to zero.

Applying this lookup to every pixel creates a backprojection image:

- Bright regions match the object’s color well.

- Dark regions do not match it well.

The weight is not the probability that the pixel belongs to the mine. It only measures how common that color was inside the original box. A seabed pixel with the same color can also receive a high weight.

## Step 3: MeanShift moves the box

MeanShift begins with a fixed-size search box in the second image.

It calculates the weighted center of all matching pixels inside that box:

$$
\begin{aligned}
x_c &= \frac{\sum_{x,y} x\,w(x,y)}{\sum_{x,y} w(x,y)} \\
y_c &= \frac{\sum_{x,y} y\,w(x,y)}{\sum_{x,y} w(x,y)}
\end{aligned}
$$

Here:

- The function w at coordinates x and y gives the color-matching weight.

- The horizontal-center variable gives the weighted horizontal center.

- The vertical-center variable gives the weighted vertical center.

Pixels with large weights pull the center toward themselves more strongly.

### A small one-dimensional example

Suppose three matching pixels are found at horizontal positions:

$$
x = 2,\ 3,\ 4
$$

Their weights are:

$$
w = 64,\ 255,\ 255
$$

The weighted center is:

$$
\begin{aligned}
x_c &= \frac{2(64) + 3(255) + 4(255)}{64 + 255 + 255} \\
    &= \frac{1913}{574} \approx 3.33
\end{aligned}
$$

The strongest matching pixels are toward the right, so MeanShift moves the box’s center toward:

$$
x_c \approx 3.33
$$

After moving the box, MeanShift calculates the weighted center again. It repeats this process until the box moves by less than the configured threshold or reaches the maximum number of iterations.

The box changes position during MeanShift, but its size remains fixed.

## Step 4: CamShift measures the spread

CamShift first runs MeanShift to find the object’s approximate center. It then measures how widely the matching pixels are distributed around that center.

The horizontal weighted variance is:

$$
\sigma_x^2 = \frac{\sum_{x,y} w(x,y)(x-x_c)^2}{\sum_{x,y} w(x,y)}
$$

The vertical weighted variance is:

$$
\sigma_y^2 = \frac{\sum_{x,y} w(x,y)(y-y_c)^2}{\sum_{x,y} w(x,y)}
$$

These quantities measure horizontal and vertical spread:

- Small variance means the matching pixels are tightly concentrated.

- Large variance means they are spread over a larger region.

CamShift also calculates the weighted covariance:

$$
\sigma_{xy} = \frac{\sum_{x,y} w(x,y)(x-x_c)(y-y_c)}{\sum_{x,y} w(x,y)}
$$

This measures whether the matching pixels form a diagonal pattern.

The three values form a covariance matrix:

$$
\Sigma = \begin{bmatrix}
\sigma_x^2 & \sigma_{xy} \\
\sigma_{xy} & \sigma_y^2
\end{bmatrix}
$$

The eigenvectors of this matrix provide the object’s main directions. The eigenvalues describe the spread along those directions.

If the eigenvalues are:

$$
\lambda_1 \quad\text{and}\quad \lambda_2
$$

OpenCV calculates the approximate box dimensions as:

$$
\begin{aligned}
\text{length} &= 4\sqrt{\lambda_1} \\
\text{width} &= 4\sqrt{\lambda_2}
\end{aligned}
$$

Thus:

- A wide distribution produces a larger box.

- A concentrated distribution produces a smaller box.

- A diagonal distribution produces a rotated box.

## More matching pixels do not automatically mean a larger box

This is an important detail.

Suppose every weight becomes ten times larger. The horizontal variance becomes:

$$
\sigma_x^2 = \frac{\sum 10w(x,y)(x-x_c)^2}{\sum 10w(x,y)}
$$

The factor of ten appears in both the numerator and denominator, so it cancels:

$$
\sigma_x^2 = \frac{\sum w(x,y)(x-x_c)^2}{\sum w(x,y)}
$$

Therefore, increasing every weight does not make the box larger.

The box size depends on the spatial spread of the weights, not simply their total number or magnitude.

For example, consider two equally weighted matching pixels.

In the first case, they are close together:

$$
x = 3,\ 4
$$

Their center is:

$$
x_c = 3.5
$$

Their variance is:

$$
\sigma_x^2 = \frac{(3-3.5)^2 + (4-3.5)^2}{2} = 0.25
$$

Therefore:

$$
\sigma_x = 0.5
$$

In the second case, the same number of matching pixels is spread farther apart:

$$
x = 1,\ 6
$$

The center is still:

$$
x_c = 3.5
$$

But the variance is now:

$$
\sigma_x^2 = \frac{(1-3.5)^2 + (6-3.5)^2}{2} = 6.25
$$

Therefore:

$$
\sigma_x = 2.5
$$

Both examples have two matching pixels with the same total weight. The second distribution produces a much larger box because the pixels are farther apart.

## Does CamShift repeatedly resize the box?

Within one OpenCV CamShift call:

1. MeanShift repeatedly moves a fixed-size window until its center stabilizes.

2. CamShift measures the final weighted distribution.

3. It calculates a new position, size, and rotation once.

4. It returns the resulting rotated box.

CamShift does not repeatedly resize the box until its dimensions stabilize during a single call.

Instead, continuous size adaptation happens across images:

```text
Frame 1 box
    ↓
CamShift on frame 2
    ↓
New position and size
    ↓
Use that box for frame 3
    ↓
New position and size
```

If the object appears larger in the next image, its matching pixels should occupy a wider region. The measured spread increases, so CamShift returns a larger box.

If the object appears smaller, its matching pixels should be more concentrated. The measured spread decreases, so CamShift returns a smaller box.

In the labeling tool, this process happens once whenever the user applies CamShift to the next image. The returned box becomes the starting object box for the following image.

Finally, the tool converts CamShift’s rotated box into an axis-aligned bounding box. A strongly rotated object may therefore receive a slightly looser final box because the axis-aligned rectangle must enclose the entire rotated result.

OpenCV describes histogram backprojection in its [backprojection documentation](https://docs.opencv.org/5.0/tutorials/imgproc/histograms/back_projection/back_projection.html), and the precise resizing calculations appear in the [OpenCV CamShift implementation](https://github.com/opencv/opencv/blob/4.x/modules/video/src/camshift.cpp).
