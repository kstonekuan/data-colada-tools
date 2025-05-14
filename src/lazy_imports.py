#!/usr/bin/env python3
"""
Lazy import utility for data forensics tools.
This module provides lazy-loaded versions of heavy dependencies
to improve startup time.
"""

# Global dictionary to hold lazy-loaded modules
_modules = {}


def get_seaborn():
    """Lazily import seaborn only when needed"""
    if "seaborn" not in _modules:
        import seaborn as sns

        _modules["seaborn"] = sns
    return _modules["seaborn"]


def get_scipy():
    """Lazily import scipy modules only when needed"""
    if "scipy" not in _modules:
        import scipy

        _modules["scipy"] = scipy
    return _modules["scipy"]


def get_scipy_stats():
    """Lazily import scipy.stats only when needed"""
    if "scipy.stats" not in _modules:
        from scipy import stats

        _modules["scipy.stats"] = stats
    return _modules["scipy.stats"]


def get_statsmodels():
    """Lazily import statsmodels only when needed"""
    if "statsmodels" not in _modules:
        import statsmodels.api as sm

        _modules["statsmodels"] = sm
    return _modules["statsmodels"]


def get_pypdf2():
    """Lazily import PyPDF2 only when needed"""
    if "PyPDF2" not in _modules:
        import PyPDF2

        _modules["PyPDF2"] = PyPDF2
    return _modules["PyPDF2"]


def get_pyreadstat():
    """Lazily import pyreadstat only when needed"""
    if "pyreadstat" not in _modules:
        import pyreadstat

        _modules["pyreadstat"] = pyreadstat
    return _modules["pyreadstat"]
