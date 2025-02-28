from setuptools import setup, find_packages

setup(
    name="unified-generative-metrics",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn"
    ],
    author="Alexis Fox",
    author_email="alexis.fox@duke.edu",
    description="Information-theoretic metrics for generative model evaluation",
    url="https://github.com/NSSAC/PrecisionRecallMetric",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
