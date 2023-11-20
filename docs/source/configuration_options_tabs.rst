.. tab-set::

   .. tab-item:: geometry

      Defines the turbine's geometric parameters.

      .. list-table::
         :widths: 20 80
         :header-rows: 0

         * - **Mandatory**
           - True
         * - **Valid types**
           - ``dict``


      .. tab-set::

         .. tab-item:: cascade_type

            Specifies the types of cascade of each blade row.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``
               * - **Valid options**
                 - ``stator``, ``rotor``

         .. tab-item:: radius_hub

            Hub radius at the inlet and outlet of each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: radius_tip

            Tip radius at the inlet and outlet of each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: pitch

            Blade pitch (aslo known as spacing) for each cascade

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: chord

            Blade chord for each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: stagger_angle

            Blade stagger angle for each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: opening

            Blade opening for each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: diameter_le

            Leading-edge diameter for each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: wedge_angle_le

            Wedge angle at the leading edge of each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: metal_angle_le

            Metal angle at the leading edge of each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: metal_angle_te

            Metal angle at the trailing edge of each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: thickness_te

            Trailing edge thickness of the blades for each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: thickness_max

            Maximum thicknesses of the blades for each cascade.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: tip_clearance

            Tip clearance of the blades for each cascade (usually zero for stator blades).

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``

         .. tab-item:: throat_location_fraction

            Defines the position of the throat in the blade passages as a fraction of the cascade's axial length. This parameter is relevant when the annulus shape varies from the inlet to the outlet of the cascade, due to factors like flaring or non-constant radius. A value of 1 indicates that the throat is located exactly at the exit plane, aligning the throat's area and radius with the exit plane's dimensions. Adjusting this fraction allows for precise modeling of the throat location relative to the exit.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``list``, ``ndarray``
   .. tab-item:: operation_points

      Defines operating conditions for turbine performance analysis. This can be provided in two formats. The first format is as a list of dictionaries, where each dictionary defines a single operation point. The second format is as a single dictionary where each key has a single value or an array of values. In this case, the function internally generates all possible combinations of operation points, similar to creating a performance map, by taking the Cartesian product of these ranges.

      .. list-table::
         :widths: 20 80
         :header-rows: 0

         * - **Mandatory**
           - True
         * - **Valid types**
           - ``dict``, ``list``, ``ndarray``


      .. tab-set::

         .. tab-item:: fluid_name

            Name of the working fluid.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``str``

         .. tab-item:: T0_in

            Stagnation temperature at the inlet. Unit [K].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: p0_in

            Stagnation pressure at the inlet. Unit [Pa].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: p_out

            Static pressure at the exit. Unit [Pa].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: omega

            Angular speed. Unit [rad/s].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: alpha_in

            Flow angle at the inlet. Unit [deg].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``
   .. tab-item:: performance_map

      Specifies a range of operating conditions for creating the turbine's performance map. This option is expected to be a dictionary where each key corresponds to a parameter (like inlet pressure, angular speed, etc.) and its value is a scalar or an array of possible values for that parameter. The code generates the complete set of operation points internally by calculating all possible combinations of operating conditions (i.e., taking the cartesian product of the ranges).

      .. list-table::
         :widths: 20 80
         :header-rows: 0

         * - **Mandatory**
           - False
         * - **Valid types**
           - ``dict``


      .. tab-set::

         .. tab-item:: fluid_name

            Name of the working fluid.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``str``

         .. tab-item:: T0_in

            Stagnation temperature at the inlet. Unit [K].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: p0_in

            Stagnation pressure at the inlet. Unit [Pa].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: p_out

            Static pressure at the exit. Unit [Pa].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: omega

            Angular speed. Unit [rad/s].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``

         .. tab-item:: alpha_in

            Flow angle at the inlet. Unit [deg].

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``number``, ``ndarray``, ``list``
   .. tab-item:: model_options

      Specifies the options related to the physical modeling of the problem

      .. list-table::
         :widths: 20 80
         :header-rows: 0

         * - **Mandatory**
           - True
         * - **Valid types**
           - ``dict``


      .. tab-set::

         .. tab-item:: choking_condition

            Closure condition used to predict turbine choking.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``str``
               * - **Valid options**
                 - ``deviation``, ``mach_critical``, ``mach_unity``

         .. tab-item:: deviation_model

            Deviation model used to predict the exit flow angle at subsonic conditions.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``str``
               * - **Valid options**
                 - ``aungier``, ``ainley_mathieson``, ``zero_deviation``

         .. tab-item:: blockage_model

            Model used to predict the blockage factor due to boundary layer displacement thickness.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Default value**
                 - 0.0
               * - **Valid types**
                 - ``float``, ``str``
               * - **Valid options**
                 - ``flat_plate_turbulent``, ``<numeric value>``

         .. tab-item:: rel_step_fd

            Relative step size of the finite differences used to approximate the critical condition Jacobian.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - 0.001
               * - **Valid types**
                 - ``float``

         .. tab-item:: loss_model

            Specifies the options of the methods to estimate losses.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - True
               * - **Valid types**
                 - ``dict``


            .. tab-set::

               .. tab-item:: model

                  Name of the model used to calculate the losses.

                  .. list-table::
                     :widths: 20 80
                     :header-rows: 0

                     * - **Mandatory**
                       - True
                     * - **Valid types**
                       - ``str``
                     * - **Valid options**
                       - ``kacker_okapuu``, ``moustapha``, ``benner``, ``benner_moustapha``, ``isentropic``

               .. tab-item:: loss_coefficient

                  Definition of the loss coefficient used to characterize the losses.

                  .. list-table::
                     :widths: 20 80
                     :header-rows: 0

                     * - **Mandatory**
                       - True
                     * - **Valid types**
                       - ``str``
                     * - **Valid options**
                       - ``stagnation_pressure``

               .. tab-item:: inlet_displacement_thickness_height_ratio

                  Ratio of the endwall boundary layer displacement thickness at the inlet of a cascade to the height of the blade. Used in the secondary loss calculations of the `benner` loss model.

                  .. list-table::
                     :widths: 20 80
                     :header-rows: 0

                     * - **Mandatory**
                       - False
                     * - **Default value**
                       - 0.011
                     * - **Valid types**
                       - ``float``

               .. tab-item:: tuning_factors

                  Specifies tuning factors to have control over the weight of the different loss components.

                  .. list-table::
                     :widths: 20 80
                     :header-rows: 0

                     * - **Mandatory**
                       - False
                     * - **Valid types**
                       - ``dict``


                  .. tab-set::

                     .. tab-item:: profile

                        Multiplicative factor for the profile losses.

                        .. list-table::
                           :widths: 20 80
                           :header-rows: 0

                           * - **Mandatory**
                             - False
                           * - **Default value**
                             - 1.0
                           * - **Valid types**
                             - ``float``

                     .. tab-item:: incidence

                        Multiplicative factor for the incidence losses.

                        .. list-table::
                           :widths: 20 80
                           :header-rows: 0

                           * - **Mandatory**
                             - False
                           * - **Default value**
                             - 1.0
                           * - **Valid types**
                             - ``float``

                     .. tab-item:: secondary

                        Multiplicative factor for the secondary losses.

                        .. list-table::
                           :widths: 20 80
                           :header-rows: 0

                           * - **Mandatory**
                             - False
                           * - **Default value**
                             - 1.0
                           * - **Valid types**
                             - ``float``

                     .. tab-item:: trailing

                        Multiplicative factor for the trailing edge losses.

                        .. list-table::
                           :widths: 20 80
                           :header-rows: 0

                           * - **Mandatory**
                             - False
                           * - **Default value**
                             - 1.0
                           * - **Valid types**
                             - ``float``

                     .. tab-item:: clearance

                        Multiplicative factor for the tip clearance losses.

                        .. list-table::
                           :widths: 20 80
                           :header-rows: 0

                           * - **Mandatory**
                             - False
                           * - **Default value**
                             - 1.0
                           * - **Valid types**
                             - ``float``
   .. tab-item:: solver_options

      Specifies options related to the numerical methods used to solve the problem

      .. list-table::
         :widths: 20 80
         :header-rows: 0

         * - **Mandatory**
           - False
         * - **Default value**
           - {}
         * - **Valid types**
           - ``dict``


      .. tab-set::

         .. tab-item:: method

            Name of the numerical method used to solve the problem. Different methods may offer various advantages in terms of accuracy, speed, or stability, depending on the problem being solved

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - lm
               * - **Valid types**
                 - ``str``
               * - **Valid options**
                 - ``lm``, ``hybr``

         .. tab-item:: tolerance

            Termination tolerance for the solver. This value determines the precision of the solution. Lower tolerance values increase the precision but may require more computational time.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - 1e-08
               * - **Valid types**
                 - ``float``, ``float64``

         .. tab-item:: max_iterations

            Maximum number of solver iterations. This sets an upper limit on the number of iterations to prevent endless computation in cases where convergence is slow or not achievable.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - 100
               * - **Valid types**
                 - ``int``, ``int64``

         .. tab-item:: derivative_method

            Finite difference method used to calculate the problem Jacobian

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - 2-point
               * - **Valid types**
                 - ``str``
               * - **Valid options**
                 - ``2-point``, ``3-point``

         .. tab-item:: derivative_rel_step

            Relative step size of the finite differences used to approximate the problem Jacobian. This step size is crucial in balancing the truncation error and round-off error. A larger step size may lead to higher truncation errors, whereas a very small step size can increase round-off errors due to the finite precision of floating point arithmetic. Choosing the appropriate step size is key to ensuring accuracy and stability in the derivative estimation process.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - 0.0001
               * - **Valid types**
                 - ``float``

         .. tab-item:: display_progress

            Whether to print the convergence history to the console. Enabling this option helps in monitoring the solver's progress and diagnosing convergence issues during the solution process.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - True
               * - **Valid types**
                 - ``bool``
   .. tab-item:: general_settings

      Defines general settings controlling the behavior of the program.

      .. list-table::
         :widths: 20 80
         :header-rows: 0

         * - **Mandatory**
           - False
         * - **Valid types**
           - ``dict``


      .. tab-set::

         .. tab-item:: skip_validation

            Whether to skip the configuration validation or not.

            .. list-table::
               :widths: 20 80
               :header-rows: 0

               * - **Mandatory**
                 - False
               * - **Default value**
                 - False
               * - **Valid types**
                 - ``bool``

