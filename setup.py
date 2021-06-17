from setuptools import setup

setup(
    name='yellowcab',
    version='0.0.1dev1',
    description="Semester Project - Programming Data Science",
    author="Diem Ly, Tin Nguyen, Pratyush Nayak, Tiffany Zukas",
    author_email="student@uni-koeln.de",
    packages=["yellowcab"],
    install_requires=[
        'pandas',
        'click',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': ['yellowcab=yellowcab.cli:main']
    }
)
