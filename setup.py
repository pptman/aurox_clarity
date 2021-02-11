import setuptools

setuptools.setup(
    name="aurox_clarity",
    version="1.0.0",
    description="Control and image processing for Aurox Clarity devices",
    url="https://github.com/pptman/clarity_processor",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    install_requires=[
        "hidapi >= 0.9.0",
        "opencv-python >= 3.4.5",
    ],
)
