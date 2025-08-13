import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="handmusclemodels",                    
    version="0.1.4",
    author="Jonas Koellner",                         
    author_email="jonas.koellner@iss.uni-stuttgart.de",     
    description="Models for EMG and MMG data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/derkoellner/handmusclemodels", 
    project_urls={
        "Bug Tracker": "https://github.com/derkoellner/handmusclemodels/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["handmusclemodels", "handmusclemodels.*"]),
    python_requires='>=3.9',
    install_requires=[
        "numpy>=1.24",
        "torch>=2.7",
        "pandas>=2.2",
        "einops>=0.8",
        "wandb>=0.19",
        "tqdm>=4.0",
        "scikit-learn>=1.0",
    ],
    include_package_data=True,
    zip_safe=False,
)