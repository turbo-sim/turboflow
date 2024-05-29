# Configuration
OUTPUT_FILE = "source/nomenclature.rst"
NOMENCLATURE_DATA = {
    "Latin Symbols": [
        ("H", r":math:`H`", "Blade height", r"m"),
        ("s1", r":math:`s`", "Blade pitch", r"m"),
        ("s2", r":math:`s`", "Specific entropy", r":math:`\mathrm{J/kg\,K}`"),
        ("s3", r":math:`s_\mathrm{ax}`", "Axial spacing between cascades", r"m"),
        ("o", r":math:`o`", "Blade opening", r"m"),
        ("c", r":math:`c`", "Blade chord", r"m"),
        (
            "CR",
            r":math:`CR`",
            "Contraction ratio in the :cite:`benner_empirical_2006` loss model",
            r"--",
        ),
        ("c_a", r":math:`c_\mathrm{ax}`", "Blade axial chord", r"m"),
        ("r_m", r":math:`r_{\mathrm{m}}`", "Turbine mean radius", r"m"),
        ("d_m", r":math:`d_{\mathrm{m}}`", "Turbine mean diameter", r"m"),
        ("d_s", r":math:`d_{\mathrm{s}}`", "Turbine specific diameter", "--"),
        ("r_h", r":math:`r_{\mathrm{h}}`", "Radius at the hub of the blades", r"m"),
        ("r_t", r":math:`r_{\mathrm{t}}`", "Radius at the tip of the blades", r"m"),
        ("t", r":math:`t`", "Blade thickness", r"m"),
        ("t_te", r":math:`t_{\mathrm{te}}`", "Trailing edge thickness", r"m"),
        ("t_max", r":math:`t_{\mathrm{max}}`", "Maximum blade thickness", r"m"),
        ("v", r":math:`v`", "Absolute flow velocity", r"m/s"),
        (
            "v_0",
            r":math:`v_{0}`",
            "Isentropic velocity (also known as spouting velocity)",
            r"m/s",
        ),
        ("w", r":math:`w`", "Relative flow velocity", r"m/s"),
        ("u", r":math:`u`", "Blade velocity", r"m/s"),
        ("N", r":math:`N`", "Number of turbine stages", "--"),
        ("m", r":math:`\dot{m}`", "Mass flow rate", r"kg/s"),
        ("w", r":math:`\dot{W}`", "Actual power output", r"W"),
        ("w_s", r":math:`\dot{W}_{s}`", "Isentropic power output", r"W"),
        ("PR", r":math:`PR`", "Pressure ratio", "--"),
        ("p", r":math:`p`", "Static pressure", r"Pa"),
        ("p0", r":math:`p_{0}`", "Stagnation pressure", r"Pa"),
        ("T", r":math:`T`", "Static temperature", r"K"),
        ("T0", r":math:`T_{0}`", "Stagnation temperature", r"K"),
        ("h", r":math:`h`", "Static specific enthalpy", r"J/kg"),
        ("h0", r":math:`h_{0}`", "Stagnation specific enthalpy", r"J/kg"),
        (
            "h_ts",
            r":math:`\Delta h_{\mathrm{ts},s}`",
            "Total-to-static isentropic specific enthalpy change",
            r"J/kg",
        ),
        (
            "h_tt",
            r":math:`\Delta h_{\mathrm{tt},s}`",
            "Total-to-static isentropic specific enthalpy change",
            r"J/kg",
        ),
        ("a", r":math:`a`", "Speed of sound", r"m/s"),
        ("Y", r":math:`Y`", "Stagnation pressure loss coefficient", "--"),
        ("Cf", r":math:`C_{\mathrm{f}}`", "Diffuser skin friction coefficient", "--"),
        (
            "Ft",
            r":math:`F_t`",
            "Tangential loading parameter in the :cite:`benner_empirical_2006` loss model",
            "--",
        ),
        (
            "Zte",
            r":math:`Z_{\mathrm{te}}`",
            "Passage vortex penetration depth at the trailing edge",
            "m",
        ),
    ],
    "Greek Symbols": [
        ("01", r":math:`\alpha`", "Absolute flow angle", r":math:`^{\circ}`"),
        ("02", r":math:`\beta`", "Relative flow angle", r":math:`^{\circ}`"),
        ("02", r":math:`\gamma`", "Heat capacity ratio", "--"),
        ("04", r":math:`\delta`", "Flow deviation angle", r":math:`^{\circ}`"),
        (
            "04_0",
            r":math:`\delta`",
            "Diffuser semi-divergence angle",
            r":math:`^{\circ}`",
        ),
        ("04_1", r":math:`\delta`", "Boundary layer thickness", r"m"),
        ("04_2", r":math:`\delta^*`", "Displacement thickness", r"m"),
        (
            "04_3",
            r":math:`\delta_{\mathrm{fl}}`",
            "Blade flaring angle",
            r":math:`^{\circ}`",
        ),
        ("05", r":math:`\epsilon`", "Blade wedge angle", r":math:`^{\circ}`"),
        ("05", r":math:`\epsilon_{\mathrm{cl}}`", "Tip clearance gap", r"m"),
        ("07", r":math:`\eta_{ts}`", "Total-to-static isentropic efficiency", "--"),
        ("07", r":math:`\eta_{tt}`", "Total-to-total isentropic efficiency", "--"),
        ("08", r":math:`\theta`", "Metal angle", r":math:`^{\circ}`"),
        ("08", r":math:`\Delta \theta`", "Camber angle", r":math:`^{\circ}`"),
        ("09", r":math:`i`", "Incidence angle", r":math:`^{\circ}`"),
        ("11", r":math:`\lambda`", "Hub-to-tip radii ratio", "--"),
        ("12", r":math:`\mu`", "Dynamic viscosity", r"Pa\,s"),
        (
            "14",
            r":math:`\xi`",
            "Stagger angle (also known as setting angle)",
            r":math:`^{\circ}`",
        ),
        ("17", r":math:`\rho`", "Density", r"kg/m^3"),
        ("19", r":math:`\tau_{\mathrm{w}}`", "Wall shear stress", r"Pa"),
        ("21", r":math:`\phi`", "Flow coefficient", "--"),
        (
            "21b",
            r":math:`\phi^2`",
            r"Kinetic energy loss coefficient (:math:`\phi^2+\Delta\phi^2=1`)",
            "--",
        ),
        (
            "21b",
            r":math:`\Delta\phi^2`",
            "Complementary of the kinetic energy loss coefficient",
            "--",
        ),
        (
            "22",
            r":math:`\chi`",
            "Incidence parameter in the :cite:`moustapha_improved_1990` loss model",
            "--",
        ),
        (
            "22",
            r":math:`\chi'`",
            "Incidence parameter in the :cite:`benner_influence_1997` loss model",
            "--",
        ),
        ("23", r":math:`\psi`", "Loading coefficient", "--"),
        ("24", r":math:`\omega`", "Angular velocity", r"rad/s"),
    ],
    "Abbreviations": [
        ("E", "EOS", "Equation of State", ""),
        ("O", "ODE", "Ordinary Differential Equation", ""),
        ("C", "CFD", "Computational Fluid Dynamics", ""),
        ("S", "SQP", "Sequential Quadratic Programming (optimization algorithm)", ""),
        ("I", "IP", "Interior Point (optimization algorithm)", ""),
    ],
    "Subscripts": [
        ("m", r":math:`m`", "Meridional direction", ""),
        ("x", r":math:`x`", "Axial direction", ""),
        ("r", r":math:`r`", "Radial direction", ""),
        ("t", r":math:`\theta`", "Tangential direction", ""),
        ("in", r":math:`\mathrm{in}`", "Inlet of the cascade or turbomachine", ""),
        ("out", r":math:`\mathrm{out}`", "Outlet of the cascade or turbomachine", ""),
        ("0", r":math:`0`", "Stagnation state", ""),
        ("1", r":math:`1,2,3,\ldots`", "Flow stations", ""),
        ("s1", r":math:`s`", "Refers to isentropic quantities", ""),
        ("s2", r":math:`\mathrm{s}`", "Refers to specific quantitities", ""),
        (
            "rel",
            r":math:`\mathrm{rel}`",
            "Relative to the rotating frame of reference",
            "",
        ),
        ("error", r":math:`\mathrm{error}`", "Violation of an equality constraint", ""),
        ("inc", r":math:`\mathrm{inc}`", "Incidence loss", ""),
        ("p", r":math:`\mathrm{p}`", "Profile loss", ""),
        ("sec", r":math:`\mathrm{sec}`", "Secondary loss", ""),
        ("te", r":math:`\mathrm{te}`", "Trailing edge", ""),
        ("cl", r":math:`\mathrm{cl}`", "Tip clearance", ""),
        ("mid", r":math:`\mathrm{mid}`", "Blade midspan", ""),
    ],
}

# Greek alphabet order
# 1.  Alpha (Α, α)
# 2.  Beta (Β, β)
# 3.  Gamma (Γ, γ)
# 4.  Delta (Δ, δ)
# 5.  Epsilon (Ε, ε)
# 6.  Zeta (Ζ, ζ)
# 7.  Eta (Η, η)
# 8.  Theta (Θ, θ)
# 9.  Iota (Ι, ι)
# 10. Kappa (Κ, κ)
# 11. Lambda (Λ, λ)
# 12. Mu (Μ, μ)
# 13. Nu (Ν, ν)
# 14. Xi (Ξ, ξ)
# 15. Omicron (Ο, ο)
# 16. Pi (Π, π)
# 17. Rho (Ρ, ρ)
# 18. Sigma (Σ, σ)
# 19. Tau (Τ, τ)
# 20. Upsilon (Υ, υ)
# 21. Phi (Φ, φ)
# 22. Chi (Χ, χ)
# 23. Psi (Ψ, ψ)
# 24. Omega (Ω, ω)


def generate_rst_table(nomenclature_data):

    # Table header
    rst_output = []
    rst_output.append(".. _nomenclature:")
    rst_output.append("")
    rst_output.append("Nomenclature")
    rst_output.append("============")
    rst_output.append("")
    rst_output.append(".. list-table::")
    rst_output.append("   :widths: 20 70 10")
    rst_output.append("   :header-rows: 1")
    rst_output.append("   :class: nomenclature-center")
    rst_output.append("")
    rst_output.append("   * - Symbol")
    rst_output.append("     - Description")
    rst_output.append("     - Unit")

    # Loop through each category
    for category, entries in nomenclature_data.items():
        # Sort entries based on the first element (sorting key)
        sorted_entries = sorted(entries, key=lambda x: x[0].strip().lower())

        # Add the category header
        rst_output.append("")
        rst_output.append(f"   * - **{category}**")
        rst_output.append("     - ")
        rst_output.append("     - ")

        # Add the entries
        for entry in sorted_entries:
            _, symbol, description, unit = entry
            rst_output.append(f"   * - {symbol}")
            rst_output.append(f"     - {description}")
            rst_output.append(f"     - {unit}")

    return "\n".join(rst_output)


if __name__ == "__main__":
    with open(OUTPUT_FILE, "w") as file:
        file.write(generate_rst_table(NOMENCLATURE_DATA))
    print(f"Nomenclature generation completed! Check file: {OUTPUT_FILE}")


# def generate_rst_table(nomenclature_entries, group_name):
#     header = f"""
# {group_name}
# {'-' * len(group_name)}

# .. list-table::
#    :widths: 20 80 20
#    :header-rows: 1

#    * - Symbol
#      - Description
#      - Unit
# """

#     # Sort entries by their sorting key
#     sorted_entries = sorted(nomenclature_entries, key=lambda x: x[0])

#     entries_rst = ""
#     for entry in sorted_entries:
#         # Exclude the sorting key when generating the RST table
#         _, symbol, description, unit = entry
#         entries_rst += f"   * - {symbol}\n"
#         entries_rst += f"     - {description}\n"
#         entries_rst += f"     - {unit}\n"

#     return header + entries_rst

# if __name__ == "__main__":
#     latin_symbols = [
#         ["A", ":math:`A`", "Area", "m2"],
#         ["v", ":math:`v`", "Absolute velocity", "m/s"],
#         ["w", ":math:`v`", "Relative velocity", "m/s"]
#     ]

#     greek_symbols = [
#         ["A", ":math:`\\alpha`", "Absolute flow angle measured from axial direction", "degree or rad"],
#         ["B", ":math:`\\beta`", "Relative flow angle measured from axial direction", "degree or rad"],
#         ["R", ":math:`\\rho`", "Fluid density", "kg/m3"]
#     ]

#     subscripts = [
#         ["s", ":math:`s`", "Isentropic", ""]
#     ]

#     rst_output = generate_rst_table(latin_symbols, "Latin Symbols")
#     rst_output += generate_rst_table(greek_symbols, "Greek Symbols")
#     rst_output += generate_rst_table(subscripts, "Subscripts")
#     print(rst_output)
