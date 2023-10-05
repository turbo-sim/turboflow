import subprocess

def run_sphinx_apidoc(output_dir, src_dir):
    cmd = ["sphinx-apidoc", "-o", output_dir, src_dir, '-e']
    subprocess.check_call(cmd)
    print("Sphinx apidoc completed successfully.")

def run_sphinx_build(docs_dir=".", build_dir="_build", builder="html"):
    cmd = ["sphinx-build", "-b", builder, docs_dir, build_dir]
    subprocess.check_call(cmd)
    print(f"Sphinx build ({builder} format) completed successfully.")

def run_generate_nomenclature(script_path):
    """Run the nomenclature generation script."""
    cmd = ["python", script_path]
    subprocess.check_call(cmd)
    print("Nomenclature generation completed successfully.")

if __name__ == "__main__":
    run_generate_nomenclature("generate_nomenclature.py")
    # run_sphinx_apidoc(output_dir="source/api/", src_dir="../meanline_axial")
    run_sphinx_build(docs_dir=".", build_dir="_build", builder="html")
