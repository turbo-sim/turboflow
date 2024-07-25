
import os
import platform

# Check OS type
SUPPORTED_OS = ["Windows", "Linux"]
os_name = platform.system()
if os_name not in SUPPORTED_OS:
    raise Exception("OS not supported")
os_name = os_name.lower()

CONFIG_DIR = os.path.join('tests', 'config_files')
REGRESSION_CONFIG_DIR = os.path.join('tests', f'regression_data_{os_name}')

# Dictionary to map test names to their respective configuration files
TEST_CONFIGS = {
    'test_performance_analysis': [
                                "performance_analysis_evaluate_cascade_throat.yaml",
                                "performance_analysis_evaluate_cascade_critical.yaml",
                                "performance_analysis_evaluate_cascade_isentropic_throat.yaml",
                                "performance_analysis_kacker_okapuu.yaml",
                                "performance_analysis_moustapha.yaml",
                                "performance_analysis_zero_deviation.yaml",
                                "performance_analysis_ainley_mathieson.yaml",
                            ]
,
    'test_design_optimization': ["design_optimization.yaml"],
    # Add more test configurations as needed
}

# Dictionary of configuration files for regression data generation
REGRESSION_CONFIGS = {
    'performance_analysis': [
                                "performance_analysis_evaluate_cascade_throat.yaml",
                                "performance_analysis_evaluate_cascade_critical.yaml",
                                "performance_analysis_evaluate_cascade_isentropic_throat.yaml",
                                "performance_analysis_kacker_okapuu.yaml",
                                "performance_analysis_moustapha.yaml",
                                "performance_analysis_zero_deviation.yaml",
                                "performance_analysis_ainley_mathieson.yaml",
                            ],
    'design_optimization': None,
}

def get_config_files(test_name):
    """
    Get the list of configuration files for a given test.
    """
    config_files = TEST_CONFIGS.get(test_name, [])
    return [os.path.join(CONFIG_DIR, f) for f in config_files]

def get_regression_config_files(test_name):
    """
    Get the list of configuration files for regression data generation.
    """
    config_files = REGRESSION_CONFIGS.get(test_name, [])
    if config_files == None:
            return None
    else:
        return [os.path.join(CONFIG_DIR, f) for f in config_files]