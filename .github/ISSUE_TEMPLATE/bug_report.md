---
name: Bug report
about: Something is broken or producing wrong results
labels: bug
---

## Describe the bug

A clear and concise description of what the bug is.

## Minimal reproducible example

```python
import numpy as np
from openfit import Fit

x = ...
y = ...
result = Fit("hill4p", x, y, weights="1/y2").run()
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback if applicable.

```
Traceback (most recent call last):
  ...
```

## Environment

- openfit version: <!-- python -c "import openfit; print(openfit.__version__)" -->
- Python version:
- OS:
- scipy version:
- numpy version:

## Additional context

Any other context about the problem (e.g. data characteristics, weight scheme used).
