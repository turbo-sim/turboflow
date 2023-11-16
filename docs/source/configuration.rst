.. _configuration:


Configuration
================

This section documents all available options for the configuration file used by the ``meanline_axial`` package.
Each entry outlines the specific parameters and constraints that govern the behavior of the package, 
providing guidance on how to correctly set up and customize your configuration.

Each entry in the file below includes the following keys:

- ``description``: Provides an explanation of the option.
- ``is_mandatory``: Indicates whether the option must be specified.
- ``default_value``: The default value used when the option is not mandatory and not specified.
- ``expected_types``: The data type(s) that are valid for the option.
- ``valid_options``: Lists permissible string values for the option. If the parameter accepts a numerical value or other types than an option string, the file below displays ``None``, indicating no restricted set of string options.
- ``_nested``: Specifies the children options of the configuration the entry.




.. literalinclude:: configuration_options.yaml
   :language: yaml
   :linenos:

