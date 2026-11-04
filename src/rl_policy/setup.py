from glob import glob

from setuptools import find_packages, setup

package_name = 'rl_policy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Akshat Guduru',
    maintainer_email='akshat.guduru@gmail.com',
    description='Phase-1 RL policy node with safety layer',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = rl_policy.policy_node:main',
            'stage0_monitor = rl_policy.stage0_monitor:main',
        ],
    },
)
