# 🏥 Health Equity Insights Platform (HEIP)
### *An Intersectional Approach to Identifying Vertical Equity Gaps*

---

## 📌 Project Overview
The **Health Equity Insights Platform** is a strategic diagnostic tool designed to analyze healthcare expenditure through a multi-dimensional lens. By intersecting clinical encounter data with socio-economic indicators—such as **Income (Wealth)**, **Race/Ethnicity**, **Gender**, and **Age**—the platform identifies specific "Critical Cohorts" that face disproportionate financial burdens.

Our goal is to move beyond simple averages and uncover the **intersectional realities** of healthcare access and cost-impact to inform better policy and resource allocation.

---

## 🎯 Key Research Questions
1. **Vertical Equity:** How does the financial burden of healthcare scale across different income tiers?
2. **Intersectionality:** How do gender and race exacerbate the cost-impact of specific chronic conditions?
3. **Geographic Intensity:** Which regions demonstrate the highest clinical costs when filtered for underserved demographics?
4. **Coverage Sufficiency:** Is existing insurance coverage adequate for the most expensive clinical conditions?

---

## 🛠️ Technology Stack
* **Language:** Python 3.12+
* **Framework:** [Streamlit](https://streamlit.io/) (Web Interface)
* **Analytics:** Pandas (Data Manipulation), Altair (Declarative Visualization)
* **Deployment:** Streamlit Cloud

---

## 📂 Repository Structure
```text
health_care/
├── app/
│   └── main.py              # Main Application Logic & UI
├── data/
│   ├── patients.csv         # Demographic & Socio-economic Metadata
│   └── encounters*.csv      # Clinical Encounter & Cost Records
├── requirements.txt         # Project Dependencies
└── README.md                # Project Documentation
