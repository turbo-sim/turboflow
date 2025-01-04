import os
import yaml
import scipy.io as io
import numpy as np
import turboflow as tf


# Directories
input_dir = 'files_mat'
output_dir = 'files_yaml'

# Define name mappings and desired order
name_mappings = {
    "casename": "casename",
    "media": "fluid_name",
    "Ptin": "p0_in",
    "Ttin": "T0_in",
    "G_beta": "mass_flow_PR",
    "N_beta": "RPM_PR",
    "beta_tt": "PR_tt",
    "G_eta": "mass_flow_eta",
    "N_eta": "RPM_eta",
    "eta_tt": "eta_tt",
    "partial_geometry": "partial_geometry"
}

# Custom Dumper for YAML to improve readability
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to convert .mat file to .yaml
def convert_mat_to_yaml(input_path, output_path):

    # Load .mat file
    mat_data = io.loadmat(input_path)

    # Remove metadata keys that start with '__'
    mat_data = {key: value for key, value in mat_data.items() if not key.startswith('__')}

    # Manually handle partial_geometry by squeezing extra dimensions
    if 'partial_geometry' in mat_data:
        mat_data['partial_geometry'] = [np.squeeze(item) for item in mat_data['partial_geometry'][0][0]]

    # Rename fields based on name_mappings
    renamed_data = {}
    for old_name, new_name in name_mappings.items():
        if old_name in mat_data:
            renamed_data[new_name] = mat_data.pop(old_name)

    # Convert to scalars
    for key in ["casename", "fluid_name", "T0_in", "p0_in"]:
        renamed_data[key] = np.squeeze(renamed_data[key])

    # Convert numpy data to Python-native types with specified precision
    renamed_data = tf.convert_numpy_to_python(renamed_data, precision=16)

    # Write to YAML file with custom formatting
    with open(output_path, 'w') as yaml_file:
        yaml.dump(renamed_data, yaml_file, Dumper=MyDumper, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    
    # Iterate over all .mat files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.yaml")
            convert_mat_to_yaml(input_path, output_path)
            print(f"Converted {filename} to {output_path}")