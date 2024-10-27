from setuptools import setup, find_packages

setup(
    name="quantum_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A quantum circuit simulator implementation",
    keywords="quantum computing, simulation, quantum circuits",
    python_requires=">=3.8",
)
