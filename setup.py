from setuptools import setup, find_packages

setup(
    name="pca-image-compression",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "click==8.1.8",
        "contourpy==1.3.0",
        "cycler==0.12.1",
        "fonttools==4.56.0",
        "importlib_resources==6.5.2",
        "joblib==1.4.2",
        "kiwisolver==1.4.7",
        "matplotlib==3.9.4",
        "numpy==2.0.2",
        "packaging==24.2",
        "pillow==11.1.0",
        "pio==0.0.3",
        "pyparsing==3.2.1",
        "python-dateutil==2.9.0.post0",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        "six==1.17.0",
        "threadpoolctl==3.5.0",
        "zipp==3.21.0"
    ],
    entry_points={
        "console_scripts": [
            "pca_image_compression = pca_image_compression.main:main",
        ],
    },
)