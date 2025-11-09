flowchart TD
  A[boundary_conditions, reference_values, variables, geom_list, model_options] --> B[evaluate_axial_turbine_componentwise]
  B -->|loop per component i| C[_validate_geometry_component]
  B --> D[evaluate_cascade_inlet]
  B --> E[evaluate_cascade_exit (+losses,+blockage)]
  B --> F[cm.evaluate_choking]
  B --> G{next is rotor?}
  G -- yes & current is stator --> H[evaluate_cascade_interspace]
  G -- no --> I[pass exit → next inlet]
  B --> J[tf.combine_to_dict_of_arrays]
  J --> K[compute_stage_performance_componentwise]
  J --> L[compute_overall_performance_componentwise]
  B --> M[residuals incl. outlet p error]

Function Dependency Tree
evaluate_axial_turbine_componentwise
├─ _validate_geometry_component
├─ evaluate_cascade_inlet
│  ├─ evaluate_velocity_triangle_in
│  │  └─ math.sind / math.cosd / math.arctand
│  ├─ fluid.get_state (HmassSmass_INPUTS)
│  └─ utils.add_string_to_keys
├─ evaluate_cascade_exit
│  ├─ evaluate_velocity_triangle_out
│  │  └─ math.sind / math.cosd / math.arctand
│  ├─ fluid.get_state (HmassSmass_INPUTS, PSmass_INPUTS)
│  ├─ compute_blockage_boundary_layer
│  └─ lm.evaluate_loss_model
├─ cm.evaluate_choking
├─ evaluate_cascade_interspace
│  ├─ math.arctand
│  └─ fluid.get_state (DmassHmass_INPUTS)
├─ tf.combine_to_dict_of_arrays
├─ compute_stage_performance_componentwise
└─ compute_overall_performance_componentwise

# Optional (not in main path unless enabled)
evaluate_cascade_throat
├─ evaluate_velocity_triangle_out
├─ fluid.get_state
├─ compute_blockage_boundary_layer
└─ lm.evaluate_loss_model


sequenceDiagram
  participant Vars as variables (scaled)
  participant In as evaluate_cascade_inlet
  participant Ex as evaluate_cascade_exit
  participant Ch as cm.evaluate_choking
  participant Int as evaluate_cascade_interspace
  participant Next as next component

  Vars->>In: {h0_in, s_in, alpha_in, v_in}
  In-->>In: velocity_triangle_in + thermo + Re/Ma + ṁ
  In->>Ex: {rothalpy, beta_out_i, w_out_i, s_out_i}
  Ex-->>Ex: velocity_triangle_out + thermo + Re/Ma + ṁ(blockage) + losses
  Ex->>Ch: choking_input + inlet_plane + exit_plane + model_options
  Ch-->>Ex: critical residuals + critical state
  Ex->>Int: if (stator→rotor): (h0_out, v_m, v_t, ρ, A, r, blockage)
  Int-->>Next: {h0_in, s_in, alpha_in, v_in} for rotor


