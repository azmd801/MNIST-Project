import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "mnist-Project"
AUTHOR_USER_NAME = "azhar"
AUTHOR_EMAIL = "azmd801@gmail.com"



setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python package for end to MINIST Classification using MLflow",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".")
)