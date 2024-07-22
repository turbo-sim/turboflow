# Import packages
import os
import sys

# Define package name for re-usability
package_name = "pysolver_view"

# Define bash command to append cwd to PYTHONPATH
PACKAGE_PATH = os.getcwd()
bashrc_header = f"# Append {package_name} package to PYTHONPATH"
if sys.platform == 'win32': # Windows
    bashrc_line = f'export PYTHONPATH=$PYTHONPATH\;"{PACKAGE_PATH}"'
else:  # Linux or MacOS
    bashrc_line = f'export PYTHONPATH=$PYTHONPATH:"{PACKAGE_PATH}"'

# Locate the .bashrc file
bashrc_path = os.path.expanduser("~/.bashrc")

# Ask for user confirmation with default set to no
response = input(f"Do you want to add the {package_name} path to your .bashrc? [yes/NO]: ")
if response.lower() in ['y', 'yes']:
    try:
        # Check if the line already exists in the .bashrc
        with open(bashrc_path, 'r') as file:
            if bashrc_line in file.read():
                print(f".bashrc already contains the {package_name} path.")
            else:
                with open(bashrc_path, 'a') as file_append:
                    file_append.write(f"\n{bashrc_header}")
                    file_append.write(f"\n{bashrc_line}\n")
                print(f"Path to {package_name} package added to .bashrc")
                print(f"Restart the terminal or run 'source ~/.bashrc' for the changes to take effect.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("Operation aborted by user.")

