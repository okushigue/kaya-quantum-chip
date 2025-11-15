from setuptools import setup, find_packages

setup(
    name="kaya-quantum-chip",
    version="1.0.0",
    description="Kaya Quantum Photonic Chip: Noise-Enhanced Quantum Computing",
    author="Jeff",
    author_email="okushigue@gmail.com",
    url="https://github.com/okushigue/kaya-quantum-chip",
    packages=find_packages(),
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-aer>=0.12.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "chaos>=0.1.2",
        "nolds>=0.5.2",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gpu": [
            "tensorflow-gpu>=2.12.0",
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "full": [
            "tensorflow>=2.12.0",
            "torch>=2.0.0",
            "xgboost>=1.7.0",
            "plotly>=5.13.0",
            "bokeh>=3.0.0",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Quantum Computing",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)