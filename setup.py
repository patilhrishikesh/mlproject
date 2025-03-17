from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->list[str]:
    '''
    This function read s the requirements from the file and returns a list of requirements'''
    requirement = []
    HYPEN_E_DOT = '-e .'
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace('\n', '') for req in requirement]
        
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
    
    return requirement    

setup(
    name='mlproject',
    version='0.0.1',
    author='Hrishikesh',
    author_email='patilarjun415@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
    
)
