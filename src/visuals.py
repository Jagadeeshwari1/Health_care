import streamlit as st

def show_cost_analysis(report):
    st.subheader("💰 Cost vs Income Analysis")
    st.bar_chart(report.set_index("CITY")[["HEALTHCARE_EXPENSES", "INCOME"]])

def show_city_analysis(df, city):
    st.subheader(f"📍 Insights for {city}")
    city_df = df[df['CITY'] == city]

    if city_df.empty:
        st.warning("No data available")
        return

    st.write(city_df[['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']].describe())
