import setuptools
from pathlib import Path

root_path = Path(__file__).parent
version_file = root_path / "VERSION.txt"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kamui",
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
    setuptools_git_versioning={
        "enabled": True,
        "version_file": version_file,
        "count_commits_from_version_file": True,
        "dev_template": "{tag}.{branch}{ccount}",
        "dirty_template": "{tag}.{branch}{ccount}",
    },
    setup_requires=["setuptools-git-versioning<2"],
)
