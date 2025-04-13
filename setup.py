from setuptools import setup, find_packages

setup(
    name="data-forensics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "openpyxl",
        "statsmodels"
    ],
    entry_points={
        'console_scripts': [
            'data-forensics=src.main:main',
        ],
    },
)