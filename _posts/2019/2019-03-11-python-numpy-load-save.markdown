---
layout: post
title: Python - Numpy File Management
date: 2019-03-11 13:19
subtitle: Load, Save, npy, npz
comments: true
header-img: img/post-bg-2015.jpg
tags:
  - Python
---
## `numpy.save()` saves npy file to cache

When this runs:

```python
np.save(npy_path, points)
```

Linux usually does not immediately send every byte to the physical SSD. It first places the file’s data in RAM, in the **page cache**:

```
NumPy array
    ↓ np.save()
Linux page cache in RAM
    ↓ eventually
Physical SSD
```

Then you immediately run:

```python
loaded = np.load(npy_path)
```

Linux notices that the file’s contents are still in its page cache:

```
np.load()
    ↓
Linux page cache in RAM
    ↓
New NumPy array
```

As you load more data from npy/npz, Linux will cache data pages, but in the meantime may evict some. So

## `npy` is faster than `npz` and not much bigger

`npy` stores data on disk contiguously, this allows faster loading using `self.points = np.load(points_path,mmap_mode="r")`.

I staged a test to measure:

- Array: (2000000, 4), dtype=float32
- Raw size: 30.52 MiB

And I found:

```
Format                  Size MiB     Save ms    Cold load ms
------------------------------------------------------------
NPY                        30.52        6.87           14.91
NPZ uncompressed           30.52       17.55           26.55
NPZ compressed             28.61      807.26          129.24
```

Core Code snippet:

```python
import gc
import os
import statistics
import time


def evict_from_page_cache(path):
    fd = os.open(path, os.O_RDONLY)

    try:
        # Ensure data written by np.save() has reached storage.
        os.fsync(fd)

        # Ask Linux to remove this file's pages from the OS page cache.
        os.posix_fadvise(
            fd,
            0,
            0,
            os.POSIX_FADV_DONTNEED,
        )
    finally:
        os.close(fd)


def median_cold_load(path, is_npz):
    times = []

    for _ in range(REPEATS):
        gc.collect()
        evict_from_page_cache(path)

        start = time.perf_counter()

        if is_npz:
            with np.load(path, allow_pickle=False) as archive:
                loaded = archive["points"]
        else:
            loaded = np.load(path, allow_pickle=False)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        del loaded

    return statistics.median(times)
```

### npy normal loading vs mmap loading

Assuming we have 5000 point clouds as our dataset. If we save them in 1 npy file:

```python
points = np.load("points.npy", mmap_mode="r",)
```

Normal loading is

```
Open → read all 4 GB → allocate array → begin training → Afterward, every access is ordinary RAM access.
```

`mmap` always reads the header of the array, create a memory view to each point cloud, read points from disk into cache directly.  

```
Open NPY:
    read one header and establish mmap

Select frame:
    create a view; usually no data read yet

First use:
    missing file pages → SSD → page cache

Second use:
    cached file pages → RAM access

Memory pressure:
    Linux may discard pages → future access returns to SSD
```

- Normal loading can be faster when you use the complete dataset during every epoch and it comfortably fits in RAM. (So if your dataset is small, this is a good fit)
- Mmap is usually faster overall when you use only part of the file. Mmap can be slower during the first epoch if access is highly random. After the file becomes cached, mmap performance may approach normal RAM access, though page-table and indexing overhead remains.
