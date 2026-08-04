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
        # 0.35 is where Wan's residual up block landed, and Wan's blocks are the only ones any
        # module here imports at import time. The QwenImage, HunyuanVideo and LTX-2 families are
        # resolved through distvae.modules.adapters.diffusers_blocks instead, so an install too
        # old for one of them keeps every other adapter and is told which class it lacks only if
        # it tries to shard that VAE. Raising this floor for them would cost more than it buys.
        install_requires=["torch>=2.2", "diffusers>=0.35.0", "transformers"],
        extras_require={
            "dev": [
                "pytest",
                "black",
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
