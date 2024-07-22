---
title: 'TurboFlow: Meanline Modelling of Turbomachinery'
tags:
  - python
  - turbomachinery
  - meanline modeling
  - design optimization
  - performance analysis
authors:
  - name: Lasse B. Anderson
    orcid: 0009-0001-2293-9954
    corresponding: true
    affiliation: 1
  - name: Roberto Agromayor
    orcid: 0009-0009-9289-1130
    affiliation: 2
  - name: Fredrik Haglind
    orcid: 0000-0003-4086-8325
    affiliation: 2
  - name: Lars O. Nord
    orcid: 0000-0002-2734-5821
    affiliation: 1
affiliations:
  - name: Department of Energy and Process Engineering, NTNU - Norwegian University of Science and Technology, Trondheim, Norway
    index : 1
  - name: Department of Civil and Mechanical Engineering, Technical University of Denmark, Kongens Lyngby, Denmark
    index: 2

date: 26 June 2024
bibliography: paper.bib
---

# Summary

TurboFlow is a Python package for meanline modeling of turbomachinery, providing a comprehensive framework for on- and off-design performance analysis and design optimization. It employs an equation-oriented model formulation, making it compatible with gradient-based equation solvers and optimization algorithms for efficient computations. The package features a modular architecture that allows for seamless integration of various submodels, enabling users to 
select and combine different models for calculating losses, flow angles and choking, tailoring the analysis to specific needs. The strcuture also facilitate for users to implement other submodels for these purposes. 
TurboFlow provides access to advanced equations of state for real gas fluid properties by interfacing to the CoolProp library. The accuracy and computational robustness of the implemented models have been demonstrated through comprehensive validation against experimental data.  

TurboFlow comes with comprehensive documentation, including installation guides, tutorials, detailed model descriptions and a complete API reference. 
This extensive resource ensures that users can easily learn how to use the package and apply it effectively in their projects. For more details, visit the [documentation pages](https://turbo-sim.github.io/TurboFlow/).
Additionally, the package includes preconfigured examples that demonstrate performance analysis and design optimization. These examples serve as practical guides and starting point for users to apply TurboFlow to their own projects. 

The package source code is hosted in a [Github repository](https://github.com/turbo-sim/TurboFlow) [@turboflow]. Through Github Actions, an automated test suite is included, which checks the functionality of the performance analysis and design optimization, as well as all submodels. It enables continuous integration, ensuring that code changes are systematically tested and validated. This comprehensive testing framework provides confidence that the code works as expected, maintaining the reliability of the package with each update.

With these features, TurboFlow should present a reliable and flexible tool for researchers and engineers within the field of turbomachinery. 

# Statement of need

Meanline models are essential for simulation of turbomachinery [@dixon_fluid_2014]. For design processes, they enable rapid evaluation of design concepts
and are used to establish key geometrical parameters. The preliminary design forms the basis for subsequent refined design steps, and is crucial for achieving high-efficiency turbomachinery [@macchi_organic_2017]. Furthermore, meanline models offer a method to quickly, yet accurately, predict performance, making them well-suited for system-level analyses involving turbomachines, both at design and off-design conditions.

Despite the importance of these models, there is no established reference meanline tool for turbomachinery modeling available. Altough there are several commercial tools available:

- CFturbo [@noauthor_cfturbo_nodate]
- AxSTREAM [@noauthor_axstream_nodate]
- TURBOdesign Suite [@noauthor_turbodesign_nodate]
- Concepts NREC [@noauthor_concepts_nodate]

these are closed source, limiting the ability to modify, extend, or debug the models. 

Several meanline models developed in academic settings also suffer from being closed source:

- zTurbo from TU-delft [@pini_preliminary_2013]
- axTur from Politecnico di Milano [@macchi_organic_2017]
- OTAC from Nasa Glenn research Center [@hendricks_meanline_2016]

The few open-source meanline models that do exist come with significant limitations in terms of programming language, model formulation and restricted functionality. These models and their features are summarized in the following table: 

| Reference                          | Year | Programming language | Model formulation     | Functionalities      | 
|------------------------------------|------|----------------------|-----------------------|----------------------|
| [@genrup_reduced-order_2005]       | 2005 | MATLAB (proprietary) | Sequential            | Performance analysis |
| [@denton_multallopen_2017]         | 2017 | FORTRAN77 (legacy)   | Lack of documentation | Design optimization  |
| [@agromayor_preliminary_2019]      | 2019 | MATLAB (proprietary) | Equation-oriented     | Design optimization  |
| [@brind_data-driven_2024]          | 2024 | Python               | Equation-oriented     | Design optimization  |

The use of diverse programming languages, such as MATLAB and FORTRAN77, presents accessibility and compatibility issues. MATLAB-based models are proprietary, which limit their accessibility to those with MATLAB licenses.
While legacy languages like FORTRAN77 might be more accessible, they fall short in terms of modern features and extensive community support. Consequently, models developed with these languages are less efficient to develop and less attractive to potential contributors, hampering development and collaboration. Furthermore, models adopting a sequential model formulation, solve sets of model equations sequentially through multiple nested iterations. This approach can lead to unreliable convergence and prolonged execution times due to the high number of model equations evaluations required. In contrast, an equation-oriented model formulation solves a larger set of equations simultaneously, enhancing reliability and computational efficiency. Lastly, the functionalities provided by existing models differ, with some focusing solely on performance analysis and others on design optimization, yet no single open-source model offers a comprehensive solution for both. 

TurboFlow addresses these gaps with a robust, open-source framework for meanline turbomachinery modeling. It combines performance analysis and design optimization within a flexible, modular 
architecture, accommodating various submodels seamlessly. This flexibility allows for the integration of new submodels, giving users the options to tailor the analysis for their application. 
The model adopts an equation-oriented formulation, allowing integration with gradient-based solvers and offering the potential for faster convergence compared to methods based on the sequential model formulation.
TurboFlow’s open source framework enables other researchers and industry practitioners to use and contribute to its development, positioning it as the first community-driven effort in 
turbomachinery meanline modeling. Through a collaborative effort, TurboFlow can be expanded with new models and features, having a significant potential to advance turbomachinery meanline modeling.

# Acknowledgment
The research leading to the presented work has received funding from the EEA/Norway Grants and the Technology Agency of the Czech Republic within the KAPPA Program. 
Project code TO01000160. This work also received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie 
grant agreement No 899987. The financial support is gratefully acknowledged.

# References
