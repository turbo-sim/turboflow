.. _validation:

Model validation
==============================

The purpose of this section is to demonstrate the accuracy and relibility in predicting axial turbines performance. For this purpose, three cases with qualitatively different behavior are selected. 
This includes two single-stage turbines, where one experiences choking in the stator :cite:`kofskey_turbine_1974`, and the other in the rotor :cite:`kofskey_design_1972`, and a two-stage turbine, to show the capability of the model to
simulate multistage turbine configurations :cite:`kofskey_design_1972`:

   - :ref:`choked_stator`
   - :ref:`choked_rotor`
   - :ref:`two_stage`

The validation was perfomed in terms of simulating mass flow rate, torque and turbine exit absolute flow angle, at a range of total-to-static pressure ratios and angular speeds.
For each parameter, the simulated value was compared to the measured value, and a table is presented for each parameter to quantify the fraction of point within different uncertainty limits. 
In addition, the simulated and experimental vlaues are compared graphically. 

The setup simulation and solver options used for the validation is reported below:

.. code-block:: python

   simulation_options: 
      deviation_model : aungier  
      choking_model : evaluate_cascade_critical 
      rel_step_fd: 1e-4 
      loss_model:  
         model: benner 
         loss_coefficient: stagnation_pressure
     
   performance_analysis :
      solver_options: 
         method: hybr  
         tolerance: 1e-8  
         max_iterations: 100  
         derivative_method: "2-point" 


.. _choked_stator:

One-stage choked stator
-------------------------

Mass flow rate
^^^^^^^^^^^^^^^^
.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<5.0%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%

.. image:: ./images/validation/one_stage_stator/1974_mass_flow_rate.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/one_stage_stator/error_1974_mass_flow_rate_error.png
..    :width: 40%

Torque
^^^^^^^^
.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 22.2%
     - 27.3%
     - 33.3%
     - 14.3%
     - 25.6%
   * - :math:`<5.0%`
     - 33.3%
     - 63.6%
     - 66.6%
     - 57.1%
     - 56.4%
   * - :math:`<10%`
     - 88.8%
     - 72.7%
     - 83.3%
     - 85.7%
     - 82.0%

.. image:: ./images/validation/one_stage_stator/1974_torque.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/one_stage_stator/error_1974_torque_error.png
..    :width: 40%

Absolute flow angle
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 66.6%
     - 81.8%
     - 100.0%
     - 100.0%
     - 87.8%
   * - :math:`<5.0%`
     - 88.8%
     - 100.0%
     - 100.0%
     - 100.0%
     - 97.6%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%

.. image:: ./images/validation/one_stage_stator/1974_absolute_flow_angle.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/one_stage_stator/error_1974_absolute_flow_angle_error.png
..    :width: 40%

.. _choked_rotor:

One-stage choked rotor
-------------------------
Mass flow rate
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<5.0%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%

.. image:: ./images/validation/one_stage_rotor/1972_mass_flow_rate.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/one_stage_rotor/error_1972_mass_flow_rate.png
..    :width: 40%

Torque
^^^^^^^^

.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 27.3%
     - 76.9%
     - 100.0%
     - 100.0%
     - 77.1%
   * - :math:`<5.0%`
     - 72.7%
     - 100.0%
     - 100.0%
     - 100.0%
     - 93.8%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%

.. image:: ./images/validation/one_stage_rotor/1972_torque.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/one_stage_rotor/error_1972_torque.png
..    :width: 40%

Absolute flow angle
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 60.0%
     - 100.0%
     - 100.0%
     - 40.0%
     - 74.4%
   * - :math:`<5.0%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 80.0%
     - 94.9%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
.. image:: ./images/validation/one_stage_rotor/1972_absolute_flow_angle.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/one_stage_rotor/error_1972_absolute_flow_angle.png
..    :width: 40%

.. _two_stage:

Two-stage 
-----------

Mass flow rate
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<5.0%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%


.. image:: ./images/validation/two-stage/1972_2stage_mass_flow_rate.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/two-stage/error_1972_2stage_mass_flow_rate.png
..    :width: 40%

Torque
^^^^^^^^
.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 53.8%
     - 92.3%
     - 76.9%
     - 57.1%
     - 69.9%
   * - :math:`<5.0%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%

.. image:: ./images/validation/two-stage/1972_2stage_torque.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/two-stage/error_1972_2stage_torque.png
..    :width: 40%

Absolute flow angle
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10, 10, 10, 10, 10, 10

   * - Error
     - 70% of :math:`\omega_\mathrm{design}`
     - 90% of :math:`\omega_\mathrm{design}`
     - 100% of :math:`\omega_\mathrm{design}`
     - 110% of :math:`\omega_\mathrm{design}`
     - Overall
   * - :math:`<2.5%`
     - 100.0%
     - 100.0%
     - 84.6%
     - 86.7%
     - 92.3%
   * - :math:`<5.0%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
   * - :math:`<10%`
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%
     - 100.0%


.. image:: ./images/validation/two-stage/1972_2stage_absolute_flow_angle.png
   :scale: 15%
   :align: center
   
.. .. image:: ./images/validation/two-stage/error_1972_2stage_absolute_flow_angle.png
..    :width: 40%
