from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wall_follow_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # THIS LINE FIXES YOUR LAUNCH FILE ISSUE
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kit',
    maintainer_email='kit@todo.todo',
    description='Wall follow package',
    license='TODO',
    tests_require=['pytest'],

    # THIS LINE FIXES YOUR NODE EXECUTION
    entry_points={
        'console_scripts': [
            'wall_follow_node = wall_follow_pkg.wall_follow_node:main',
        ],
    },
)
