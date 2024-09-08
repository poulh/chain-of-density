from setuptools import setup, find_packages

setup(
    name='Chain of Density Summarizer',
    version='1.0',
    description='Use chain of density prompt engineering pattern for better summarization',
    author='Poul Hornsleth',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'openai',
        'python-dotenv',
        'pydantic',
    ],
    entry_points={
        'console_scripts': [
            'cod = cod:main',
        ],
    },
)
