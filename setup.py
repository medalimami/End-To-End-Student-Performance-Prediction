from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    """
    This function will return list of requirements
    """
    requirements=[]
    try:
        with open(file_path) as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            requirements=[lib.replace("\n","") for lib in lines]
            if "-e ." in requirements: requirements.remove("-e .")
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirements

setup(
    name='Students-Performance',
    version='0.0.1',
    author='medali',
    author_email='dalymami741@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements('requirements.txt'),
)