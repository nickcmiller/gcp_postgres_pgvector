from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="gcp_postgres_pgvector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Nick Miller",
    author_email="miller.nick.c@gmail.com",
    description="A package for working with pgvector in Google Cloud SQL PostgreSQL",
    long_description="A comprehensive package for using pgvector with Google Cloud SQL PostgreSQL, including database operations, connection management, and utility functions.",
    url="https://github.com/nickcmiller/gcp_postgres_pgvector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)