# Meanline Axial

## Installation instructions

If you are using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html), you can use the following [Bash](https://gitforwindows.org/) command to create a new virtual environment with all the dependencies required to run the code in this repository:
``` bash
conda env create --file environment.yaml
```
This will create the `meanline_env` virtual environment and install all the packages in the specified in the YAML file.

To activate the virtual environment use:
``` bash
conda activate meanline_env
```
If you need to install additional packages you can use the following command:
``` bash
conda install <name of the package>
```
You can also install new packages by adding their names to the `environment.yaml` file and updating the environment (using `--prune` removes any dependencies that are no longer required):
``` bash
conda env update --file environment.yaml --prune
```



## To-do list
- [ ] Verify torque and efficiency deviation match
- [ ] Check if the correlation for incidence losses from Moustapha 1990 is better
- [ ] Check if baseline kacker-okapuu + moustapha 1990 is better
- [ ] Create clean dataset including torque/efficiency/flow angle for each pressure ration/speed
- [ ] Extend validation Kofskey1972 to exit flow angle
- [ ] Add x=y comparison plots for the validation
- [ ] Verify the displacement thickness value for kofskey1974
- [ ] Try to extract the shock loss at the inlet form the profile loss to have a more clear split of the losses
- [x] Add better smoother for Mach number constraint
- [ ] Add generic smoothing functions for min/max/abs/piecewise
- [ ] Replace all non-differentiable functions of the code (specially in the loss models)
- [ ] Improve initial guess computation for supersonic case (call the function `initialize()`)
- [ ] Validate model with experimental data
- [ ] Add automatic testing (pytest) to ensure that new code pushes do not introduce breaking changes
- [ ] Make function to plot performance maps more general
- [ ] Add functionality to export/import all the parameters corresponding to a calculated operating point
- [ ] Make the plotting of performance maps more general and add functionality to save figures
- [x] Add environment.yaml file to install all dependencies at once
- [x] Add Sphinx documentation structure
- [ ] Add pygmo solvers to the pysolverview interface
- [ ] Implement design optimization
  - [ ] Single-point
  - [ ] Multi-point
- [ ] Add CI/CD pipeline to run tests and update documentation on new pushes to main branch
- [ ] Think of a nice catchy name for the project
  - MeanFlow?
  - MeanStream?
  - MeanTurbo
  - MeanTurboPy
  - TurboCompuCore
  - TurboCore
  - Meanpy
  - Others?



## CI/CD

[Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)

