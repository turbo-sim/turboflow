import os
import sys
import yaml

package_path = os.path.abspath("..")

if package_path not in sys.path:
    sys.path.append(package_path)

import meanline_axial as ml


if __name__ == "__main__":
    config_options = ml.replace_types(ml.configuration_options)
    with open("source/configuration_options.yaml", "w") as file:
        yaml.dump(config_options, file, default_flow_style=False, sort_keys=False)
