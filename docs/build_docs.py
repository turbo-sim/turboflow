import subprocess


def run_sphinx_apidoc(output_dir, src_dir,exclude = None, force = False):
    cmd = ["sphinx-apidoc", "-o", output_dir, src_dir]

    # Exclude certain packages/modules
    if isinstance(exclude, list):
        for exclude_item in exclude:
            cmd.append(f"{src_dir}/{exclude_item}")

    cmd.append("-e") # put documentation for each module on its own page
    cmd.append('--no-toc') # Do not create a table of contents file
    cmd.append("-M") # Module first

    if force:
        cmd.append("-f") # Force overwriting og any existing generated files

    subprocess.check_call(cmd)
    print("Sphinx apidoc completed successfully.")


def run_sphinx_build(docs_dir=".", build_dir="_build", builder="html"):
    cmd = ["sphinx-build", "-b", builder, docs_dir, build_dir]
    subprocess.check_call(cmd)
    print(f"Sphinx build ({builder} format) completed successfully.")


def run_script(script_path):
    """Run the nomenclature generation script."""
    cmd = ["python", script_path]
    subprocess.check_call(cmd)

if __name__ == "__main__":

    exclude = ["cycles", "turbo_configurations", "add_to_pythonpath.py"]

    # run_script("build_nomenclature.py")
    # run_script("build_bibliography.py")
    # run_script("build_configuration.py")
    run_sphinx_apidoc(output_dir="source/api/", src_dir="../turboflow", exclude = exclude, force = False)
    run_sphinx_build(docs_dir=".", build_dir="_build", builder="html")
