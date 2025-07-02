# tests/test_globals.py
import pytest
from p_pack import globals

def test_globals_exist():
    """
    Test that global constants are defined and have expected types.
    """
    assert hasattr(globals, 'num_steps')
    assert isinstance(globals.num_steps, int)

    assert hasattr(globals, 'reupload_freq')
    assert isinstance(globals.reupload_freq, int)

    assert hasattr(globals, 'num_modes_circ')
    assert isinstance(globals.num_modes_circ, int)

def test_globals_values():
    """
    Test that global constants have plausible values.
    These are just sanity checks, actual values depend on your design.
    """
    assert globals.num_steps > 0
    assert globals.reupload_freq > 0
    assert globals.num_modes_circ > 0
    # Ensure reupload_freq is not greater than num_steps if used as a frequency
    assert globals.reupload_freq <= globals.num_steps

