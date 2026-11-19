from setuptools import find_packages, setup

setup(
    name="car_tests", version="0.1.0", packages=find_packages(exclude=["test"]),
    data_files=[("share/ament_index/resource_index/packages", ["resource/car_tests"]),
                ("share/car_tests", ["package.xml", "README.md"])],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="Akshat Guduru", maintainer_email="akshat.guduru@gmail.com",
    description="RB-gated vehicle measurement maneuvers and policy guard", license="MIT",
    entry_points={"console_scripts": [
        "grip_circle = car_tests.node:grip_main",
        "straight_runs = car_tests.node:straight_main",
        "steer_steps = car_tests.node:steer_main",
        "policy_guard = car_tests.node:policy_main",
    ]},
)
