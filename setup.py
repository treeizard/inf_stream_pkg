from setuptools import find_packages, setup

package_name = 'inf_stream_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['model/best.pt'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tao',
    maintainer_email='yide.tao@monash.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = inf_stream_pkg.inf_pub:main',
            'viewer = inf_stream_pkg.inf_sub:main'
        ],
    },
)
