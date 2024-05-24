#!/usr/bin/env python3
import pytest
import test_performance_analysis

# Define the list of tests
tests_list = [
            # "test_performance_analysis.py",
                "test_design_optimization.py"]

# Run pytest when the python script is executed
pytest.main(tests_list)
