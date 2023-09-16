import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kamui",
    version="0.1.1.dev",
    author="Chin-Yun Yu",
    author_email="chin-yun.yu@qmul.ac.uk",
    description="A Python package for phase unwrapping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/kamui",
    packages=["kamui"],
    install_requires=["numpy", "scipy"],
    extras_require={
        "extra": ["PyMaxflow"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
