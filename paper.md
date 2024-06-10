<!-- ---
title: 'TurboFlow: Meanline Modelling of Axial Turbines'
tags:
  - python
  - turbomachinery
  - [Roberto: Meanline modeling]
  - preliminary design [Roberto: I think design optimization sounds better]
  - performance analysis
authors:
  - name: Lasse B. Anderson
    orcid: 0009-0001-2293-9954
    affiliation: 1
  - name: Roberto Agromayor
    orcid: [Roberto: 0009-0009-9289-1130]
    affiliation: 2
  - name: Lars O. Nord
    orcid: 
    affiliation: 1
affiliations:
  - name: Department of Energy and Process Engineering, Norwegian University of Science and Technology, Trondheim, Norway
    index : 1
  - name: Department of Civil and Mechanical Engineering, Technical University of Denmarj, Kgs. Lyngby, Denmark
    index: 2

date: 6 June 2024
bibliography: paper.bib -->

# Summary

TurboFlow is a python package designed for meanline modeling of axial turbines, providing a comprehensive framework for performance analysis and design optimization. 
It employs an equation-oriented model formulation, making it compatible with gradient-based root solvers and optimization algorithms for efficient and accurate computations.
The package features a modular architecture that allows for seamless integration of various submodels.  A set of predefined submodules are provided, enabling users to 
select and combine different models for calculating losses, deviation, and choking, tailoring the analysis to specific needs. TurboFlow also integrates the CoolProp library, 
which provides advanced equations of state for real-gas fluid analysis. The model accuracy and computational robustness have been demonstrated through comprehensive validation 
against experimental data. 

TurboFlow comes with comprehensive documentation, including installation guides, tutorials and detailed model descriptions. 
This extensive resource ensures that users can easily learn how to use the package and apply it effectively in their projects. For more details, visit the [documentation pages](https://turbo-sim.github.io/TurboFlow/).
Additionally, the package includes preconfigured examples that demonstrate performance analysis and design optimization. These examples serve as practical guides and starting points for users to 
apply TurboFlow to their specific turbine projects. 

The package can be found in a [github repository](https://github.com/turbo-sim/TurboFlow) [@turboflow]. Through Github Actions, an automated test suite is included, which tests the functionality of performance analysis and design optimization, as well as all submodels, on both Windows and Linux operating systems. It enables continuous integration, 
ensuring that code changes are systematically tested and validated. This comprehensive testing framework provides confidence that the code works as expected, maintaining the reliability of 
the package with each update.

Although currently focused on axial turbines, TurboFlow is structured to be easily extended to other types of turbomachinery. This design flexibility allows for future enhancements and broader 
application across different turbomachinery components.

# Statement of need

Meanline models are essential for simulation of turbomachinery [@dixon_fluid_2014]. For design processes, they enable rapid evaluation of design concepts, allowing engineers to quickly assess the feasibility and potential 
performance of different configurations. Additionally, these models are integral to preliminary design, where key geometric parameters are defined, essential for further design process steps [@macchi_organic_2017]. 
Furthermore, meanline models offers a method to quickly but accurately predict performance, making them well-suited for system-level analysis. Combined with their potential for 
off-design performance prediction, the meanline model presents an invaluable tool for prediction of turbomachinery performance in power plants, especially as the importance of flexibility and 
part-load operation increases due to the growing interest in power production from renewable resources [@rua_optimal_2020].

Despite their importance, there is no established reference tool for turbomachinery modeling. There are several commercial tools available:

- [Cfturbo](https://cfturbo.com/)
- [AxSTREAM](https://www.softinway.com/software/)
- [TURBOdesign Suite](https://www.adtechnology.com/products)
- [Concepts NREC](https://www.conceptsnrec.com/turbomachinery-software-solutions)

However, these are closed source, limiting the ability to modify, extend, or debug the models. 

Several meanline models developed in academic settings also suffer from being closed source:

- zTurbo from TU-delft [@pini_preliminary_2013]
- axTur from Politecnico di Milano [@macchi_organic_2017]
- OTAC from Nasa Glenn research Center [@hendricks_meanline_2016]

The few open-source meanline models that do exist come with significant limitations in terms of programming language and restricted functionality. 
The opens source meanline models and their limitation are summarized in the following table: 

| Reference                          | Year | Limitation                                       |
|------------------------------------|------|--------------------------------------------------|
| [@genrup_reduced-order_2005]       | 2005 | Based on proprietary language (MATLAB)           |
| [@denton_multallopen_2017]         | 2017 | Based on legacy programming language (FORTRAN77) |
| [@agromayor_preliminary_2019]      | 2019 | Only design optimization                         |


Existing models also typically use a sequential model formulation, solving sets of equations sequentially through multiple nested iterations. This approach can lead to unreliable convergence and prolonged execution 
times due to the numerous equation evaluations required. In contrast, an equation-oriented model formulation solves a larger set of equations simultaneously, enhancing reliability and computational efficiency.

TurboFlow addresses these gaps with a robust, open-source framework for meanline turbomachinery modeling. It combines performance analysis and design optimization within a flexible, modular 
architecture, accommodating various submodels seamlessly. This flexibility allows for the integration of new submodels, giving users the options to tailor the analysis for their application. 
TurboFlow’s open source framework enables other researchers and industry practitioners to use and contribute to its development, positioning it as the first community-driven effort in 
turbomachinery meanline modeling. Through collaboration, TurboFlow can be expanded with new models and configurations, significantly advancing the meanline model field.

# Acknowledgment
The research leading to these results has received funding from the EEA/Norway Grants and the Technology Agency of the Czech Republic within the KAPPA Program. 
Project code TO01000160. This work also received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie 
grant agreement No 899987. The financial support is gratefully acknowledged.

# References
