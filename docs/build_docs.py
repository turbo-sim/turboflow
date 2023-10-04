import subprocess

def run_sphinx_apidoc(output_dir, src_dir):
    """
    Run the sphinx-apidoc command to generate API documentation.

    Parameters:
    - source_dir (str): Directory where the source files reside.
    - output_dir (str): Directory to which the API documentation should be output.
    - module_path (str): Path to the module that should be documented.
    - extensions (list of str, optional): List of extensions to pass to the sphinx-apidoc command. 
                                          Default is ['-e'].

    Raises:
    - RuntimeError: If the sphinx-apidoc command fails.
    - FileNotFoundError: If the sphinx-apidoc command is not found.
    """
    try:
        # Define the command and its arguments in a list and run the command
        cmd = ["sphinx-apidoc", "-o", output_dir, src_dir, '-e']
        subprocess.check_call(cmd)
        print("Sphinx apidoc completed successfully.")

    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to run sphinx-apidoc.")

    except FileNotFoundError:
        raise RuntimeError("'sphinx-apidoc' not found. Ensure Sphinx is installed.")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")


def run_sphinx_build(docs_dir=".", build_dir="build", builder="html"):
    """
    Run sphinx-build to generate the documentation.

    Parameters:
    - source_dir (str): Directory containing the .rst and other source files.
    - build_dir (str): Directory where the output will be written.
    - builder (str): The output format (e.g., "html", "latex").

    Raises:
    - RuntimeError: If sphinx-build fails or other errors occur.
    """
    try:
        # Define the command and its arguments in a list and run the command
        cmd = ["sphinx-build", "-b", builder, docs_dir, build_dir]
        subprocess.check_call(cmd)
        print(f"Sphinx build ({builder} format) completed successfully.")

    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to run sphinx-build for {builder} format.")

    except FileNotFoundError:
        raise RuntimeError("'sphinx-build' not found. Ensure Sphinx is installed.")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # run_sphinx_apidoc(output_dir="source/api/", src_dir="../meanline_axial")
    run_sphinx_build(docs_dir=".", build_dir="_build", builder="html")
