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
        packages=find_packages(),
        install_requires=["torch>=2.2", "diffusers>=0.27.2", "transformers"],
        url="https://github.com/PipeFusion/DistVAE.",
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
