---
layout: default
title: Installation
---

# Installation

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

---

## Install from PyPI

```bash
pip install privatecredit
```

## Install with Development Dependencies

```bash
pip install privatecredit[dev]
```

## Install from Source

```bash
git clone https://github.com/Digital-AI-Finance/private-credit.git
cd private-credit
pip install -e .
```

## Install with All Optional Dependencies

```bash
pip install privatecredit[all]
```

---

## GPU Support

For GPU acceleration, ensure you have CUDA installed and install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install privatecredit
```

---

## Verify Installation

```python
import privatecredit
print(privatecredit.__version__)
```

---

[Back to Home](index.html)
