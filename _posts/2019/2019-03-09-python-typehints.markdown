---
layout: post
title: Python - Type hints
date: 2019-03-09 13:19
subtitle: annotation
comments: true
header-img: img/post-bg-2015.jpg
tags:
  - Python
---

## `from __future__ import annotations`

A rough mental model:

- **Without it**: annotations are evaluated immediately at definition time
- **With it**: annotations are stored as strings and resolved later

### The Problem

Without `from __future__ import annotations`, forward references fail because the name isn't defined yet:

```python
class Node:  
    def __init__(self, next: Node | None = None):  # NameError: 'Node' is not defined
        self.next = next
```

This fails in older Python versions because `Node` is used before the class is fully defined.

### The Fix

```python
from __future__ import annotations

class Node:  
    def __init__(self, next: Node | None = None):  
        self.next = next
```

With this import, Python stores annotations as deferred strings (`"Node | None"`) instead of evaluating them immediately at function/class definition time. `Node | None` is kept as annotation data first, not resolved immediately, so the forward reference works.
