from setuptools import find_packages, setup

setup(
    name='gym_adaptive_management',
    packages=find_packages(include=['gym_adaptive_management', 'gym_adaptive_management.*']),
    version='0.1.0',
    description='Python library gathering examples of adaptive management problems as gymnasium environments',
    author='Luz V. Pascal',
    install_requires=["stable_baselines3","gymnasium","numpy","pandas"],
    package_data={'gym_adaptive_management' :['gym_adaptive_management/data/*']},
    include_package_data=True
)
