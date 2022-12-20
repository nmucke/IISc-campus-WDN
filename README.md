[![Build Status](https://github.com/nmucke/IISc-campus-WDN/actions/workflows/CI.yml/badge.svg?event=push)](https://github.com/nmucke/IISc-campus-WDN/actions)

# hello-world-package

This is a simple python package template.  
It uses pip for installation, flake8 for linting, pytest for testing, and coverage for monitoring test coverage.

To use it, first create a virtual environment, and install flake8, pytest, and coverage using pip.  
The following works on Windows: 
```
py -3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install flake8 pytest coverage
```

Then, install the package, run it, and test it:
```
pip install -e .
pytest
```
