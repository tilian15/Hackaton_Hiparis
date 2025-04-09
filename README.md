Update : leaderboard_4_groupe is our best work

### Hi! Welcome to our Hackathon Project Repository 🎉

This repository contains the work we presented during the **Hi! Paris Hackathon** held from **November 30th to December 1st, 2024**. The hackathon brought together interdisciplinary teams to tackle real-world challenges using cutting-edge data science and machine learning techniques.

#### About the Hackathon:
The Hi! Paris Hackathon is a renowned event hosted by Hi! Paris, a joint initiative of HEC Paris and Institut Polytechnique de Paris. It fosters collaboration among students, researchers, and industry professionals to solve societal challenges through innovation and AI-driven solutions. Our team of six consisted of three students from ENSAE Paris and three from HEC Paris, blending technical expertise with business acumen.

---

### Project Overview:

Our project aimed to **analyze and visualize groundwater levels across France**. By combining **data visualization**, **machine learning**, and a **business-oriented approach**, we developed an innovative solution with both technical depth and practical implications.

#### Repository Contents:

1. **Python Notebooks and Scripts:**
   - **`Clean_Definitions_of_Maps.ipynb`** & **`maps2.py`**:  
     These files contain Python code to visualize a map of France showing groundwater levels based on historical data, administrative divisions (regions, departments, and municipalities), and future predictions. The visualization leverages both raw data and our machine learning models.  
     To make the tool more interactive, we implemented an interface using **Streamlit**. You can execute the final application by running:  
     ```bash
     streamlit run maps2.py
     ```

2. **Business Pitch:**
   - **`Pitch - Business Idea.pdf`**:  
     As part of the challenge, we were required to contextualize our project with a clear business perspective. This file outlines our proposed solution's market potential, use cases, and broader societal impact.

3. **Machine Learning Pipeline:**
   - **`XGBoost_saison(12).ipynb`**:  
     This notebook contains the core of our machine learning workflow. Given the **large and complex dataset** with numerous variables, we invested significant effort in **feature engineering** to extract meaningful insights.  
     We used:
     - **XGBoost**: To predict groundwater levels based on historical data.  
     - **Temporal Lag Features**: Given the time-series nature of the dataset, we added lagged variables to capture temporal dependencies and improve model accuracy.

---

### Key Contributions:
- **Technical Implementation:** Mapping and visualization using Python, Streamlit, and predictive models.
- **Business Case:** Framing our solution to address real-world challenges in water resource management.
- **Collaboration:** Merging the technical prowess of ENSAE with the strategic thinking of HEC students.

Feel free to explore the repository and reach out if you have any questions!
