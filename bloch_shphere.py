import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D projection
import matplotlib.cm as cm

st.set_page_config(layout="wide")

# -------------------------
# Icon Navigation
# -------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "visualization"

st.markdown("### 🔬 Qubit Tools - Click On the Icons Below to Interact")

col_icon1, col_icon2, col_icon3 = st.columns(3)

with col_icon1:
    if st.button("🌀 Qubit Visualization"):
        st.session_state.active_page = "visualization"

with col_icon2:
    if st.button("⚛️ Qubit–Bloch Sphere"):
        st.session_state.active_page = "bloch"

with col_icon3:
    if st.button("🟦 Hadamard Gate"):
        st.session_state.active_page = "hadamard"

# -------------------------
# Shared helper functions
# -------------------------
def clamp_input(value):
    return np.clip(value, -1.0, 1.0)

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones_like(u), np.cos(v))

def plot_bloch_with_axes(vector, color, title):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='c', alpha=0.1, edgecolor='gray')
    ax.quiver(0,0,0, vector[0], vector[1], vector[2], color=color, linewidth=3)
    # Axis markers
    ax.quiver(0,0,0, 1.05,0,0, color='r')
    ax.quiver(0,0,0, 0,1.05,0, color='g')
    ax.quiver(0,0,0, 0,0,1.05, color='b')
    ax.text(1.1, 0, 0, 'X', color='r')
    ax.text(0, 1.1, 0, 'Y', color='g')
    ax.text(0, 0, 1.15, 'Z', color='b')
    ax.text(0, 0, 1.05, '|0⟩', fontsize=9)
    ax.text(0, 0, -1.1, '|1⟩', fontsize=9)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1,1,1])
    ax.set_title(title)
    return fig

# -------------------------
# PAGE 1: Qubit Visualization
# -------------------------
if st.session_state.active_page == "visualization":
    st.title("Qubit Visualization: Cartesian Plane and Bloch Sphere")
    # Top input panel
    st.header("Enter Qubit State |ψ⟩ = a|0⟩ + b|1⟩")
    col_inputs = st.columns(4)
    a_real_input = clamp_input(col_inputs[0].number_input("Re(a)", value=1.0, step=0.1))
    a_imag_input = clamp_input(col_inputs[1].number_input("Im(a)", value=0.0, step=0.1))
    b_real_input = clamp_input(col_inputs[2].number_input("Re(b)", value=0.0, step=0.1))
    b_imag_input = clamp_input(col_inputs[3].number_input("Im(b)", value=0.0, step=0.1))

    # Compute normalized qubit
    a = a_real_input + 1j*a_imag_input
    b = b_real_input + 1j*b_imag_input
    norm = np.sqrt(abs(a)**2 + abs(b)**2)
    if norm != 0:
        a /= norm
        b /= norm

    theta = 2 * np.arccos(np.clip(abs(a), 0, 1))
    phi = np.angle(b) - np.angle(a)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Top row plots
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("Re(amplitude)")
        ax.set_ylabel("Im(amplitude)")
        ax.set_title("Cartesian Plane")
        ax.scatter(a.real, a.imag, color="blue", s=100, label="a (|0⟩)")
        ax.scatter(b.real, b.imag, color="red", s=100, label="b (|1⟩)")
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.pyplot(plot_bloch_with_axes((x,y,z), "purple", "Bloch Sphere"))
    with col3:
        st.pyplot(plot_bloch_with_axes((x,y,z), "black", "Bloch Sphere (Alt View)"))

    # Right info
    st.markdown(f"### Normalized State: |ψ⟩ = ({a.real:.2f}+{a.imag:.2f}i)|0⟩ + ({b.real:.2f}+{b.imag:.2f}i)|1⟩")
    st.markdown("""
    **Notes / Reference:**
    - a and b are complex numbers  
    - |a|² + |b|² = 1  
    - |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩  
    - θ is polar angle, φ is azimuthal angle
    """)

# -------------------------
# PAGE 2: Qubit–Bloch Sphere
# -------------------------
if st.session_state.active_page == "bloch":
    st.title("Add Qubit (via θ, φ) — Plot on Bloch Sphere")
    if "added_qubits" not in st.session_state:
        st.session_state["added_qubits"] = []

    col_add_left, col_add_right = st.columns([1, 1.2])
    with col_add_left:
        st.markdown("**Enter θ and φ (degrees)**")
        with st.form(key="add_qubit_form"):
            theta_deg = st.number_input("θ (degrees)", min_value=0, max_value=180, value=90, step=1)
            phi_deg = st.number_input("φ (degrees)", min_value=0, max_value=360, value=0, step=1)
            label = st.text_input("Label (optional)", value=f"q{len(st.session_state['added_qubits'])+1}")
            submitted = st.form_submit_button("Add Qubit")
        if st.button("Clear All Added Qubits"):
            st.session_state["added_qubits"] = []
        if submitted:
            theta_rad = np.deg2rad(theta_deg)
            phi_rad = np.deg2rad(phi_deg)
            a_add = np.cos(theta_rad/2)
            b_add = np.exp(1j*phi_rad)*np.sin(theta_rad/2)
            x_add = np.sin(theta_rad)*np.cos(phi_rad)
            y_add = np.sin(theta_rad)*np.sin(phi_rad)
            z_add = np.cos(theta_rad)
            st.session_state["added_qubits"].append({
                "theta_deg": float(theta_deg),
                "phi_deg": float(phi_deg),
                "label": label if label else f"q{len(st.session_state['added_qubits'])+1}",
                "a": complex(a_add),
                "b": complex(b_add),
                "x": float(x_add),
                "y": float(y_add),
                "z": float(z_add),
            })
            st.success(f"Added qubit {label} — θ={theta_deg}°, φ={phi_deg}°")

    with col_add_right:
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, color='lightsteelblue', alpha=0.12, edgecolor='gray')
        ax.quiver(0,0,0, 1.05,0,0, color='r')
        ax.quiver(0,0,0, 0,1.05,0, color='g')
        ax.quiver(0,0,0, 0,0,1.05, color='b')
        ax.text(1.08,0,0,'X',color='r')
        ax.text(0,1.08,0,'Y',color='g')
        ax.text(0,0,1.12,'Z',color='b')
        ax.text(0.02,0.02,1.06,'|0⟩',fontsize=10)
        ax.text(0.02,0.02,-1.12,'|1⟩',fontsize=10)
        colors = cm.get_cmap("tab10")
        for i, q in enumerate(st.session_state["added_qubits"]):
            ax.quiver(0,0,0,q["x"],q["y"],q["z"],color=colors(i%10),linewidth=2)
            ax.text(q["x"]*1.05,q["y"]*1.05,q["z"]*1.05,q["label"],fontsize=9,color=colors(i%10))
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
        ax.set_zlim([-1.1,1.1])
        ax.set_box_aspect([1,1,1])
        ax.set_title("Bloch Sphere — Added Qubit Vectors")
        st.pyplot(fig)

# -------------------------
# PAGE 3: Hadamard Gate (Updated)
# -------------------------
if st.session_state.active_page == "hadamard":
    st.title("Hadamard Gate Application")
    st.markdown("""
    Enter your qubit in the form |ψ⟩ = a|0⟩ + b|1⟩.  
    The Hadamard gate will be applied to this qubit, and both the original 
    and transformed states will be displayed side by side on Bloch spheres.
    """)

    col_a, col_b = st.columns(2)
    a_in = col_a.text_input("a (complex, e.g., 0.707+0j)", "1+0j")
    b_in = col_b.text_input("b (complex, e.g., 0+0j)", "0+0j")

    try:
        a = complex(a_in)
        b = complex(b_in)
        norm = np.sqrt(abs(a)**2 + abs(b)**2)
        if norm != 0:
            a /= norm
            b /= norm

        # Input state coordinates
        theta_in = 2*np.arccos(np.clip(abs(a),0,1))
        phi_in = np.angle(b)-np.angle(a)
        x_in = np.sin(theta_in)*np.cos(phi_in)
        y_in = np.sin(theta_in)*np.sin(phi_in)
        z_in = np.cos(theta_in)

        # Apply Hadamard: H|ψ> = (a+b)|0>/√2 + (a-b)|1>/√2
        a_h = (a+b)/np.sqrt(2)
        b_h = (a-b)/np.sqrt(2)

        # Transformed state coordinates
        theta_h = 2*np.arccos(np.clip(abs(a_h),0,1))
        phi_h = np.angle(b_h)-np.angle(a_h)
        x_h = np.sin(theta_h)*np.cos(phi_h)
        y_h = np.sin(theta_h)*np.sin(phi_h)
        z_h = np.cos(theta_h)

        st.markdown(f"**Original State:** |ψ⟩ = ({a.real:.3f}+{a.imag:.3f}i)|0⟩ + ({b.real:.3f}+{b.imag:.3f}i)|1⟩")
        st.markdown(f"**After Hadamard:** |ψ⟩ = ({a_h.real:.3f}+{a_h.imag:.3f}i)|0⟩ + ({b_h.real:.3f}+{b_h.imag:.3f}i)|1⟩")

        # Side by side plots
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_bloch_with_axes((x_in,y_in,z_in),"blue","Input Qubit"))
        with col2:
            st.pyplot(plot_bloch_with_axes((x_h,y_h,z_h),"purple","After Hadamard"))

    except Exception as e:
        st.error("Invalid input for a or b. Use complex numbers like 0.707+0j.")


# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <hr style="margin-top: 2em; margin-bottom: 0.5em">
    <div style='text-align: center; font-size: 1.1em; color: black;'>
        © 2025 Anil Kumar Gundu | 
        <a href="https://anilkumargundu.github.io" target="_blank">Home</a>
    </div>
    """,
    unsafe_allow_html=True
)
