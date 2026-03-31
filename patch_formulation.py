import re

with open('src/formulation.jl', 'r') as f:
    code = f.read()

# Replace compute_sigma, compute_tau_1, compute_tau_2, compute_dsigma_du closures
# We will inject U_ref and eps_v_sq right after c_1 = ...; c_2 = ...
# or after f = ... if c_1 is not present (like in strong_residual_u).

def replacer_strong(m):
    return m.group(1) + """
    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2 
    function compute_sigma(u_val, alpha_val)
        a_term = a_resistance(alpha_val, Re, Da)
        b_term = b_resistance(alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_term + b_term * mag_u
    end
"""

code = re.sub(r'(\n    f = isnothing\(f_custom\).*?\n)\s*function compute_sigma.*?end\n', replacer_strong, code, flags=re.DOTALL)


def replacer_weak(m):
    pre_block = m.group(1)
    return pre_block + """
    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2
    
    function compute_sigma(u_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_resistance(alpha_val, Re, Da) + b_resistance(alpha_val) * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 * (U_ref / h_val) )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + 1e-12 * (U_ref / h_val) )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 * (U_ref / h_val) )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + 1e-12 * (h_val / U_ref) )
    end\n"""

code = re.sub(r'(\n    c_1 = 4.0 \* k_deg\^4; c_2 = 2.0 \* k_deg\^2\n)\s*function compute_sigma\(u_val, alpha_val\).*?function compute_tau_2\(u_val, h_val, alpha_val\).*?end\n', replacer_weak, code, flags=re.DOTALL)


def replacer_weak_jac(m):
    pre_block = m.group(1)
    return pre_block + """
    U_ref = 1.0 # Characteristic velocity scale (maintains dimensional consistency)
    eps_v_sq = (1e-6 * U_ref)^2
    
    function compute_sigma(u_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        return a_resistance(alpha_val, Re, Da) + b_resistance(alpha_val) * mag_u
    end
    function compute_tau_1(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        a_t = a_resistance(alpha_val, Re, Da)
        b_t = b_resistance(alpha_val)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 * (U_ref / h_val) )
        return 1.0 / ( (alpha_val / τ_1_NS_val) + a_t + b_t * mag_u + 1e-12 * (U_ref / h_val) )
    end
    function compute_tau_2(u_val, h_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        τ_1_NS_val = 1.0 / ( (c_1 * ν / (h_val * h_val)) + (c_2 * mag_u / h_val) + 1e-12 * (U_ref / h_val) )
        return (h_val * h_val) / ( (c_1 * alpha_val * τ_1_NS_val) + 1e-12 * (h_val / U_ref) )
    end
    function compute_dsigma_du(u_val, du_val, alpha_val)
        mag_u = sqrt(u_val ⋅ u_val + eps_v_sq)
        b_term = b_resistance(alpha_val)
        return b_term * (u_val ⋅ du_val) / mag_u
    end\n"""

code = re.sub(r'(\n    c_1 = 4.0 \* k_deg\^4; c_2 = 2.0 \* k_deg\^2\n)\s*function compute_sigma\(u_val, alpha_val\).*?function compute_dsigma_du\(u_val, du_val, alpha_val\).*?end\n', replacer_weak_jac, code, flags=re.DOTALL)


# Now write back
with open('src/formulation.jl', 'w') as f:
    f.write(code)
