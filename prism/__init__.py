"""PRISM: Progressive Residual Iterative Structural Modeling

A novel lossless image compression codec that represents images as
parametric structural models + exact corrections, rather than as
grids of pixel values.

Key innovations:
- Image-specific reversible color decorrelation (lifting PCA)
- Multi-strategy adaptive prediction with neural mixing
- Block-level parametric field modeling
- Cross-channel structural prediction
- Adaptive range coding with 2D context modeling
"""

__version__ = "0.1.0"
