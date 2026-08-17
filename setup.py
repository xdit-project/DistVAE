from setuptools import find_packages, setup


if __name__ == "__main__":
    # with open("README.md", "r") as f:
        # long_description = f.read()
    fp = open("distvae/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="DistVAE",
        author="Jinzhe Pan",
        author_email="eigensystem1318@gmail.com",
        packages=find_packages(include=["distvae", "distvae.*"]),
        # This is the oldest dependency pair covered by compatibility CI. VAE families introduced
        # in later diffusers releases are resolved lazily and name the missing class when used.
        install_requires=["torch>=2.2", "diffusers>=0.30.3"],
        extras_require={
            "pipeline": ["transformers"],
            "dev": [
                "pytest",
                "black",
                "mdformat",
                "mdformat-gfm",
                "flake8",
                "mypy",
            ],
        },
        url="https://github.com/xdit-project/DistVAE.",
        description="DistVAE: Patch Parallelism Distributed VAE for High-Resolution image generation",
        long_description=None,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
