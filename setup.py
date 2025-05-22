from setuptools import setup, find_packages

# Read README.md for long description, if it exists
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'Python port of the R nn2poly package for transforming neural networks into polynomials.'

setup(
    name='nn2poly_py',
    version='0.1.0',
    author='AI Agent (port from R nn2poly by S. L. N. T. de S. Jayalath, M. J. Reale, and R. P. Brent)',
    author_email='example@example.com', # Placeholder
    description='Python port of the R nn2poly package for transforming neural networks into polynomials.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/example/nn2poly_py', # Placeholder
    packages=find_packages(where=".", include=['nn2poly_py', 'nn2poly_py.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Assuming MIT if original R package is, or choose appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.8', # Based on typical library compatibility
    install_requires=[
        'numpy>=1.20', # Specify reasonable minimum versions
        'sympy>=1.8',
        'scipy>=1.7',
    ],
    extras_require={
        'models': [
            'tensorflow>=2.0', # For Keras parser
            'torch>=1.8',      # For PyTorch parser
        ],
        'dev': [ # Optional: dependencies for development
            'pytest',
            'flake8',
        ]
    },
    keywords='neural network polynomial conversion symbolic computation machine learning',
)
