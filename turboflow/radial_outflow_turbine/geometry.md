# 🌀 Radial Outflow Turbine — Geometry Dictionary (Per Row)

Each row (e.g. `stator_1`, `rotor_1`) in the final `geometry` dictionary contains the following keys:

---

## 🔹 1. Identity / Input Parameters

| Key | Description | Units / Notes |
|-----|--------------|---------------|
| `cascade_type` | `"stator"` or `"rotor"` | — |
| `camberline_type` | Camberline model (e.g. `"circular_arc"`) | — |
| `N_blades` | Number of blades in the row | — |
| `r_in` | Inlet mean radius | m |
| `r_out` | Outlet mean radius | m |
| `metal_angle_in` | Metal angle at inlet (w.r.t tangential) | deg |
| `metal_angle_out` | Metal angle at outlet (w.r.t tangential) | deg |
| `blade_height_in` | Blade span (hub–tip distance) at inlet | m |
| `blade_height_out` | Blade span (hub–tip distance) at outlet | m |
| `maximum_thickness` | Maximum blade thickness | m |
| `trailing_edge_thickness` | Trailing edge thickness | m |
| `maximum_thickness_location_fraction` | Axial / camberline location of max thickness | – |
| `leading_edge_radius` | Leading edge nose radius | m |
| `trailing_edge_wedge` | Trailing edge wedge angle | deg |
| `theta0` | Reference inlet circumferential angle (assumed 0) | deg |

---

## 🔹 2. Camberline Geometry

| Key | Description | Units |
|-----|--------------|-------|
| `chord` | Blade chord length | m |
| `stagger_angle` | Blade stagger angle | deg |
| `theta` | Circumferential coordinate array *(optional)* | rad |
| `d_theta` | Total change in circumferential coordinate | rad |

---

## 🔹 3. Pitch Parameters

| Key | Description | Units |
|-----|--------------|-------|
| `pitch_in` | Blade pitch at inlet (`2π·r_in / N_blades`) | m |
| `pitch_out` | Blade pitch at outlet (`2π·r_out / N_blades`) | m |
| `pitch_mean` | Pitch at mean radius (`2π·r_mean / N_blades`) | m |

---

## 🔹 4. Throat Parameters

| Key | Description | Units |
|-----|--------------|-------|
| `throat_location_fraction` | Fraction of meridional distance for throat | – |
| `r_throat` | Radius at throat (`r_in + f·(r_out – r_in)`) | m |
| `height_throat` | Blade span at throat (linear interpolation) | m |
| `throat_opening` | Minimum passage width (`pitch_out·cos(...)`) | m |
| `throat_area` | Flow area at throat (`throat_opening·height_throat`) | m² |

---

## 🔹 5. Radii (Hub Reference Only)

| Key | Description | Units |
|-----|--------------|-------|
| `radius_hub_in` | Inlet hub (mean) radius | m |
| `radius_hub_out` | Outlet hub (mean) radius | m |
| `radius_hub_mean` | Mean radius between inlet and outlet | m |
| `radius_hub_throat` | Radius at throat (if defined) | m |

---

## 🔹 6. Blade Heights

| Key | Description | Units |
|-----|--------------|-------|
| `height_in` | Blade height at inlet (`= blade_height_in`) | m |
| `height_out` | Blade height at outlet (`= blade_height_out`) | m |
| `height_mean` | Average blade height (`0.5·(height_in + height_out)`) | m |
| `height_throat` | Blade height at throat (if defined) | m |

---

## 🔹 7. Flow Areas

| Key | Description | Formula | Units |
|-----|--------------|----------|-------|
| `A_in` | Flow area at inlet | `2π·r_in·height_in` | m² |
| `A_out` | Flow area at outlet | `2π·r_out·height_out` | m² |
| `A_throat` | Flow area at throat | `2π·r_throat·height_throat` | m² |

---

## 🔹 8. Derived Geometry & Ratios

| Key | Description | Formula | Units |
|-----|--------------|----------|-------|
| `meridional_chord` | Chord projected on meridional plane | `chord·cos(stagger)` | m |
| `aspect_ratio` | Span-to-chord ratio | `height_mean / chord` | – |
| `pitch_chord_ratio` | Pitch-to-chord ratio | `pitch_mean / chord` | – |
| `solidity` | Chord-to-pitch ratio | `chord / pitch_mean` | – |
| `maximum_thickness_chord_ratio` | Maximum thickness ratio | `maximum_thickness / chord` | – |
| `trailing_edge_thickness_opening_ratio` | TE thickness / throat opening | `trailing_edge_thickness / throat_opening` | – |
| `leading_edge_diameter_chord_ratio` | 2×LE radius / chord | `2·leading_edge_radius / chord` | – |
| `flaring_angle` | Inclination of mean spanline | `atan((height_out – height_in) / meridional_chord)` | deg |

---

## 🔹 9. Optional / Machine-Level

| Key | Description | Units |
|-----|--------------|-------|
| `gauging_angle` | (To be defined later, e.g. mean flow or gauging plane angle) | deg |
| `tip_clearance_height_ratio` | Tip clearance / height_mean | – |
| `number_of_cascades` | Total cascades (rows) in YAML | – |
| `number_of_stages` | Total stages (rotor–stator pairs or rotors) | – |

---

## 🧭 Notes for Radial Outflow Convention

1. **Hub = Mean Radius Convention**  
   - Only "hub" keys are used (no "shroud" or "tip" naming).

2. **Blade Height Definition**  
   - `height_in` and `height_out` come directly from YAML.  
   - `height_mean` represents the trapezium’s mean span (average of the two).

3. **Flow Area Definitions**  
   - Areas are computed using local radius × corresponding blade height.  
   - Example: `A_in = 2π·r_in·height_in`

4. **Throat Quantities**  
   - Computed only if `throat_location_fraction` is provided.  
   - If missing, set `r_throat` and `height_throat` to `None`.

5. **Angle Units**  
   - Input metal angles: **degrees** (as in YAML).  
   - Internal trigonometric relations: **radians**.

6. **Machine-Level Parameters**  
   - `number_of_cascades = len(cfg["geometry"])`  
   - `number_of_stages` depends on your definition (e.g. #rotors or rotor–stator pairs).

---

✅ **Summary:**  
This table serves as a complete reference for all geometric parameters in the radial outflow turbine model.  
It aligns with the turbomachinery convention for radial-outflow configurations and the YAML-based geometry pipeline.
