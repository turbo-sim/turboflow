.. _configuration:


Configuration
================

This section provides a complete guide to all configuration options available within the ``meanline_axial``.
Understanding these settings is essential to utilize the software to its full potential.
Each configuration option is listed with detailed descriptions and specifications in an interactive tabbed environment.
This format allows you to navigate throat the different settings easily, while keeping, while keeping an overview of the hierarchy of options. 
Below is a summary of the key attributes for each configuration option:



.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``is_mandatory``
     - Indicates whether the option must be specified or is optional.
   * - ``default_value``
     - The default value used when the option is not mandatory and not specified.
   * - ``expected_types``
     - The data type(s) that are valid for the option.
   * - ``valid_options``
     - Lists permissible string values for the option.


Options as tabs
---------------


.. include:: configuration_options_1.rst


Options as dropdowns
---------------------

.. include:: configuration_options_2.rst




.. Alternatively, you can click the buttom below to exand the complete list of configuration options in YAML format.
.. Show the list of configuration options in YAML format


.. .. toggle:: 

..    .. literalinclude:: configuration_options.yaml
..       :language: yaml
..       :linenos:



.. - ``description``: Provides an explanation of the option.
.. - ``is_mandatory``: Indicates whether the option must be specified.
.. - ``default_value``: The default value used when the option is not mandatory and not specified.
.. - ``expected_types``: The data type(s) that are valid for the option.
.. - ``valid_options``: Lists permissible string values for the option. If the parameter accepts a numerical value or other types than an option string, the file below displays ``None``, indicating no restricted set of string options.
.. - ``_nested``: Specifies the children options of the configuration the entry.


