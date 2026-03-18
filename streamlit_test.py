import streamlit as st

st.markdown("""
<style>
body {
    background-color: #d4f8d4;
}
</style>
""", unsafe_allow_html=True)

st.title("BMI Calculator")

# Input fields
weight = st.number_input("Enter your weight (in kg):", min_value=1.0, step=0.1)
height = st.number_input("Enter your height (in cm):", min_value=1.0, step=0.1)

if st.button("Calculate BMI"):
    # Convert height from cm → meters
    height_m = height / 100

    # Formula for BMI
    bmi = weight / (height_m ** 2)

    st.write(f"### Your BMI is: **{bmi:.2f}**")

    # BMI categories
    if bmi < 18.5:
        st.warning("You are **Underweight**")
    elif 18.5 <= bmi < 24.9:
        st.success("You are **Normal weight** ✅")
    elif 25 <= bmi < 29.9:
        st.warning("You are **Overweight**")
    else:
        st.error("You are **Obese**")