from setuptools import setup, find_packages

setup(
    name="NexusAI-Suite",
    version="1.0.0",
    description="A professional, comprehensive AI/ML ecosystem.",
    author="NexusAI Contributors",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.35.0",
        "chromadb>=0.4.0",
        "fastapi>=0.104.0",
        "loguru>=0.7.0",
    ],
)
