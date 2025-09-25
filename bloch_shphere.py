import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
st.title("Qubit Visualization: Cartesian Plane and Bloch Sphere")

# --- Input Panel ---
st.header("Enter Qubit State |ψ⟩ = a|0⟩ + b|1⟩")
col_inputs = st.columns(4)

def clamp_input(value):
    return np.clip(value, -1.0, 1.0)

a_real_input = clamp_input(col_inputs[0].number_input("Re(a)", value=1.0, step=0.1))
a_imag_input = clamp_input(col_inputs[1].number_input("Im(a)", value=0.0, step=0.1))
b_real_input = clamp_input(col_inputs[2].number_input("Re(b)", value=0.0, step=0.1))
b_imag_input = clamp_input(col_inputs[3].number_input("Im(b)", value=0.0, step=0.1))

# --- Determine coefficients ---
def get_coefficients(canvas_result, a_real_input, a_imag_input, b_real_input, b_imag_input):
    if canvas_result and canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        last_obj = canvas_result.json_data["objects"][-1]
        x_click = (last_obj["left"] - 125) / 125
        y_click = -(last_obj["top"] - 125) / 125
        a_real, a_imag = np.clip(x_click, -1, 1), np.clip(y_click, -1, 1)
        b_real, b_imag = np.sqrt(max(0, 1 - (a_real**2 + a_imag**2))), 0.0
    else:
        a_real, a_imag = np.clip(a_real_input, -1, 1), np.clip(a_imag_input, -1, 1)
        b_real, b_imag = np.clip(b_real_input, -1, 1), np.clip(b_imag_input, -1, 1)

    a = a_real + 1j * a_imag
    b = b_real + 1j * b_imag
    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    if norm != 0:
        a /= norm
        b /= norm
    return a, b

# Dummy canvas_result for top figures
canvas_result = None
a, b = get_coefficients(canvas_result, a_real_input, a_imag_input, b_real_input, b_imag_input)

theta = 2 * np.arccos(np.abs(a))
phi = np.angle(b) - np.angle(a)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones_like(u), np.cos(v))

# --- Top Row: Cartesian, Simplified Bloch, Wikipedia-style Bloch ---
col1, col2, col3 = st.columns([1,1,1])

# Cartesian Plane
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
    ax.annotate(f"a=({a.real:.2f},{a.imag:.2f})", (a.real, a.imag), xytext=(5,5), textcoords="offset points", color="blue")
    ax.annotate(f"b=({b.real:.2f},{b.imag:.2f})", (b.real, b.imag), xytext=(5,5), textcoords="offset points", color="red")
    ax.legend()
    st.pyplot(fig)

# Simplified Bloch Sphere
with col2:
    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='c', alpha=0.1, edgecolor='gray')
    ax.quiver(0,0,0, 1,0,0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,1,0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,0,1, color='b', arrow_length_ratio=0.1)
    ax.quiver(0,0,0, x,y,z, color='purple', linewidth=3)
    ax.set_title("Bloch Sphere")
    st.pyplot(fig)

# Wikipedia-style Bloch Sphere θ/φ
with col3:
    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='c', alpha=0.1, edgecolor='gray')
    ax.quiver(0,0,0, 1,0,0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,1,0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0,0,0, 0,0,1, color='b', arrow_length_ratio=0.1)
    ax.quiver(0,0,0, x,y,z, color='black', linewidth=3)
    theta_vals = np.linspace(0, theta, 30)
    arc_x = np.zeros_like(theta_vals)
    arc_y = np.sin(theta_vals)
    arc_z = np.cos(theta_vals)
    ax.plot(arc_x, arc_y, arc_z, color='orange', linestyle='--', linewidth=2)
    ax.text(0.05, 0.05, np.cos(theta)/2, 'θ', color='orange', fontsize=10)
    phi_vals = np.linspace(0, phi, 30)
    arc_x = np.cos(phi_vals)
    arc_y = np.sin(phi_vals)
    arc_z = np.zeros_like(phi_vals)
    ax.plot(arc_x, arc_y, arc_z, color='magenta', linestyle='--', linewidth=2)
    ax.text(np.cos(phi)/2, np.sin(phi)/2, 0.02, 'φ', color='magenta', fontsize=10)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_title("Bloch Sphere θ/φ", fontsize=12)
    st.pyplot(fig)

# --- Normalized State ---
st.write(f"**Normalized State: |ψ⟩ = ({a:.2f})|0⟩ + ({b:.2f})|1⟩**")

# --- Bottom Row: Interactive Canvas ---
st.subheader("Interactive Canvas (click on the below plane and live interact with the bloch sphere)")
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=3,
    stroke_color="blue",
    background_color="#f9f9f9",
    update_streamlit=True,
    height=250,
    width=250,
    drawing_mode="point",
    key="canvas_bottom",
)

# --- Notes Section ---
st.markdown("""
**Notes / Reference:**

- Re(a) ∈ [0, 1], Re(b) ∈ [-1, 1], Im(b) ∈ [-1, 1], and they satisfy the normalization condition:
  [Re(a)]² + [Re(b)]² + [Im(b)]² = 1. Therefore, at most one of these three values can have absolute value 1 at any time.
- Quantum state in spherical coordinates:
  |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩
- θ ∈ [0, π] → polar angle from z-axis
- φ ∈ [0, 2π] → azimuthal angle in x-y plane
""")
st.markdown(
    """
    <hr style="margin-top: 3em; margin-bottom: 0.5em">
    <div style='text-align: center; font-size: 1.3em; color: black;'>
        © 2025 Anil Kumar Gundu | 
        <a href="https://anilkumargundu.github.io" target="_blank">Home</a>
    </div>
    """,
    unsafe_allow_html=True)

