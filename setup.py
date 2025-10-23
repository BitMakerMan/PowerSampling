"""
Setup script for Power Sampling library
"""

from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Power Sampling library for improving LLM reasoning capabilities"

try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0"
    ]

setup(
    name="reasoning-with-power-sampling",
    version="1.0.1",
    author="Power Sampling Team",
    author_email="team@powersampling.ai",
    description="Improve LLM reasoning capabilities through Power Sampling algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aakaran/reasoning-with-sampling",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "power-sampling=src.cli:main",
        ],
    },
)