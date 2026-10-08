import streamlit as st
import pickle
import numpy as np
import streamlit.components.v1 as components
import base64
import wave
import io


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EV Battery Health Prediction",
    page_icon="🔋",
    layout="wide"
)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

with open("gb_soh_model.pkl", "rb") as f:
    soh_model = pickle.load(f)

with open("gb_rul_model.pkl", "rb") as f:
    rul_model = pickle.load(f)

with open("battery_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ==========================================
# TITLE
# ==========================================

st.title("🔋 EV Battery Health Prediction Using Machine Learning")

st.write(
    "Enter the battery parameters below to predict "
    "State of Health (SOH) and Remaining Useful Life (RUL)."
)

st.divider()


# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("🔧 Battery Parameters")

col1, col2 = st.columns(2)

with col1:

    cell_voltage_avg = st.number_input(
        "Cell Voltage Average (V)",
        min_value=0.0,
        value=3.7,
        step=0.01
    )

    cell_voltage_std = st.number_input(
        "Cell Voltage Standard Deviation",
        min_value=0.0,
        value=0.05,
        step=0.01
    )

    pack_voltage = st.number_input(
        "Pack Voltage (V)",
        min_value=0.0,
        value=370.0,
        step=1.0
    )

    voltage_imbalance = st.number_input(
        "Voltage Imbalance",
        min_value=0.0,
        value=0.05,
        step=0.01
    )


with col2:

    cycle_count = st.number_input(
        "Cycle Count",
        min_value=0,
        value=500,
        step=10
    )

    capacity_fade = st.number_input(
        "Capacity Loss (%)",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    internal_resistance = st.number_input(
        "Internal Resistance",
        min_value=0.0,
        value=0.05,
        step=0.01
    )


st.divider()


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Battery Health", use_container_width=True):

    # Arrange inputs in same order used during training

    input_data = np.array([[
        cell_voltage_avg,
        cell_voltage_std,
        pack_voltage,
        voltage_imbalance,
        cycle_count,
        capacity_fade,
        internal_resistance
    ]])

    # Scale input using saved scaler
    input_scaled = scaler.transform(input_data)

    # Predictions
    soh_prediction = soh_model.predict(input_scaled)[0]
    rul_prediction = rul_model.predict(input_scaled)[0]

    # Keep values within valid ranges
    soh_prediction = np.clip(soh_prediction, 0, 100)
    rul_prediction = max(0, rul_prediction)


    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    st.subheader("📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "State of Health (SOH)",
            f"{soh_prediction:.2f}%"
        )

    with result_col2:

        st.metric(
            "Remaining Useful Life (RUL)",
            f"{rul_prediction:.0f} cycles"
        )


    # ==========================================
    # BATTERY STATUS + ALERT
    # ==========================================

    st.divider()

    st.subheader("🚨 Battery Status")


    if soh_prediction > 80:

        status = "HEALTHY"

        st.success(
            "🟢 HEALTHY\n\n"
            "Battery is in good condition."
        )

        st.info(
            "Recommendation: Continue normal operation "
            "and regular monitoring."
        )


    elif soh_prediction >= 60:

        status = "MODERATE"

        st.warning(
            "🟡 MODERATE\n\n"
            "Battery health is decreasing."
        )

        st.warning(
            "Recommendation: Monitor the battery regularly "
            "and consider maintenance."
        )


        # Soft beep
        sample_rate = 44100
        duration = 0.25
        frequency = 600

        audio = np.sin(
            2 * np.pi * frequency *
            np.linspace(
                0,
                duration,
                int(sample_rate * duration)
            )
        )

        audio = (audio * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:

            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

        audio_base64 = base64.b64encode(
            wav_buffer.getvalue()
        ).decode()

        components.html(
            f"""
            <audio autoplay>
                <source
                    src="data:audio/wav;base64,{audio_base64}"
                    type="audio/wav"
                >
            </audio>
            """,
            height=0
        )


    else:

        status = "REPLACE SOON"

        st.error(
            "🔴 REPLACE SOON\n\n"
            "Battery health is critically low."
        )

        st.error(
            "Recommendation: Battery inspection or "
            "replacement is recommended."
        )


        # Alarm sound
        sample_rate = 44100
        duration = 0.8
        frequency = 1000

        audio = np.sin(
            2 * np.pi * frequency *
            np.linspace(
                0,
                duration,
                int(sample_rate * duration)
            )
        )

        audio = (audio * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:

            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

        audio_base64 = base64.b64encode(
            wav_buffer.getvalue()
        ).decode()

        components.html(
            f"""
            <audio autoplay>
                <source
                    src="data:audio/wav;base64,{audio_base64}"
                    type="audio/wav"
                >
            </audio>
            """,
            height=0
        )


    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    st.divider()

    st.subheader("📋 Battery Health Summary")

    st.write(f"**Battery Status:** {status}")

    st.write(f"**SOH:** {soh_prediction:.2f}%")

    st.write(f"**RUL:** {rul_prediction:.0f} cycles")

    st.write(
        "The prediction is generated using the "
        "Gradient Boosting Machine Learning model."
    )