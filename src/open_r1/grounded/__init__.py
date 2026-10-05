"""Grounded question answering with verifiable rewards (additions of this fork).

This sub-package is self-contained: it does not import the upstream `open_r1`
training modules, and its rewards, data conversion and metrics run on CPU with the
standard library only. Model, dataset and trainer libraries are imported lazily by
the scripts in `scripts/grounded/`.
"""
