import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import lfilter

st.set_page_config(page_title="E-NOSE VIRTUAL LAB", page_icon="🧠", layout="wide")

st.title("🧠 E-NOSE VIRTUAL LAB")
st.markdown("### Signal Conditioning • Logic Detection • Odor Signature Visualization")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Parameters")
odor_type = st.sidebar.selectbox("Odor Type", ["Rose", "Coffee", "Alcohol", "Smoke", "Ammonia"])
concentration = st.sidebar.slider("Odor Concentration (ppm)", 1, 100, 50)
temperature = st.sidebar.slider("Temperature (°C)", 10, 50, 25)
sensor_count = st.sidebar.slider("Number of Sensors", 4, 8, 6)
threshold = st.sidebar.slider("Logic Threshold (kΩ)", 50, 200, 80)

# Signal Conditioning Controls
st.sidebar.header("🎛️ Signal Conditioning")
gain = st.sidebar.slider("Amplifier Gain", 0.5, 5.0, 2.0)
filter_alpha = st.sidebar.slider("Low-pass Filter α (0=off, 1=max filter)", 0.0, 1.0, 0.3)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.05)

run_sim = st.sidebar.button("Run Simulation")

# Sensor constants
np.random.seed(42)
alpha = np.random.uniform(0.01, 0.05, sensor_count)
beta = np.random.uniform(0.02, 0.07, sensor_count)
R0 = np.random.uniform(10, 50, sensor_count)

def sensor_response(C, T):
    """Simulate raw sensor array resistance values."""
    noise = np.random.normal(0, noise_level, sensor_count)
    R = R0 * (1 + alpha * C * np.exp(-beta * T)) + noise
    return R

def low_pass_filter(data, alpha_val):
    """Simple low-pass filter using exponential smoothing."""
    y = np.zeros_like(data)
    y[0] = data[0]
    for i in range(1, len(data)):
        y[i] = alpha_val * y[i-1] + (1 - alpha_val) * data[i]
    return y

def logic_gate_operation(R, odor):
    logic_input = R > threshold
    if odor == "Rose":
        logic_output = np.all(logic_input)
        gate = "AND"
    elif odor == "Coffee":
        logic_output = np.any(logic_input)
        gate = "OR"
    elif odor == "Alcohol":
        logic_output = np.logical_xor.reduce(logic_input)
        gate = "XOR"
    elif odor == "Smoke":
        logic_output = not np.any(logic_input)
        gate = "NOR"
    else:
        logic_output = not np.all(logic_input)
        gate = "NAND"
    return logic_input.astype(int), int(logic_output), gate

if run_sim:
    st.success(f"Simulating {odor_type} odor sensing with signal conditioning and logic analysis...")
    times = np.arange(0, 10, 0.5)
    digital_outputs = []

    col1, col2 = st.columns([2, 1])
    analog_placeholder = col1.empty()
    radar_placeholder = col1.empty()
    logic_placeholder = col2.empty()

    # Initialize memory for plotting later
    raw_data_log, conditioned_log = [], []

    for t in times:
        R_raw = sensor_response(concentration, temperature)
        R_amp = R_raw * gain
        R_filt = low_pass_filter(R_amp, filter_alpha)

        raw_data_log.append(np.mean(R_raw))
        conditioned_log.append(np.mean(R_filt))

        inputs, logic_out, gate = logic_gate_operation(R_filt, odor_type)
        digital_outputs.append(logic_out)

        # Plot Sensor Array Response
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(1, sensor_count + 1), R_raw, 'o--', label='Raw')
        ax.plot(range(1, sensor_count + 1), R_filt, 's-', label='Conditioned')
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax.set_title("Sensor Array Response (Raw vs Conditioned)")
        ax.set_xlabel("Sensor Index")
        ax.set_ylabel("Resistance (kΩ)")
        ax.legend()
        ax.grid(True)
        analog_placeholder.pyplot(fig)
        plt.close(fig)

        # Radar Chart for Odor Signature
        theta = np.linspace(0, 2 * np.pi, sensor_count, endpoint=False)
        R_radar = np.append(R_filt, R_filt[0])
        theta = np.append(theta, theta[0])
        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
        ax2.plot(theta, R_radar, color='magenta', linewidth=2)
        ax2.fill(theta, R_radar, alpha=0.25, color='magenta')
        ax2.set_title(f"{odor_type} Odor Signature Radar Plot")
        radar_placeholder.pyplot(fig2)
        plt.close(fig2)

        # Logic Gate Status Display
        logic_text = f"""
        **Logic Gate:** {gate}  
        **Inputs:** {inputs.tolist()}  
        **Output:** {logic_out}
        """
        logic_placeholder.markdown(logic_text)

        time.sleep(0.25)

    # Plot Analog Signals over Time
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(times, raw_data_log, label='Raw Average', color='gray')
    ax3.plot(times, conditioned_log, label='Conditioned Average', color='green')
    ax3.set_title("Analog Signal Conditioning Over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Resistance (kΩ)")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Digital Logic Output Plot
    fig4, ax4 = plt.subplots(figsize=(7, 2))
    ax4.step(times, digital_outputs, where='post', color='blue')
    ax4.set_ylim(-0.2, 1.2)
    ax4.set_title(f"Digital Output ({gate} Logic) Over Time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Logic Level (0/1)")
    ax4.grid(True)
    st.pyplot(fig4)

    st.success("✅ Simulation Complete!")
    st.markdown("### Observations")
    st.write("""
    - Amplifier gain boosts weak odor signals.  
    - Low-pass filter smooths out noise (higher α → slower response).  
    - Logic gate converts conditioned signals to digital decisions.  
    - Each odor type follows unique logic behavior and radar fingerprint.
    """)
else:
    st.info("👈 Adjust controls and click **Run Simulation** to start.")
