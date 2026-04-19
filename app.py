import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

clf_model = joblib.load('classification_model.pkl')
reg_model = joblib.load('regression_model.pkl')

def main():
    st.title('Student Career Prediction System')

    st.sidebar.header("About App")
    st.sidebar.info("Gunakan form di samping untuk memasukkan data. Hasil prediksi akan muncul di bagian bawah setelah tombol ditekan.")
    
    st.sidebar.subheader("Training Insight")
    fig_side, ax_side = plt.subplots(figsize=(4,3))
    features = ['Academic', 'Skills', 'Experience']
    importance = [7.99, 11.15, 17.10]
    sns.barplot(x=features, y=importance, ax=ax_side, palette='magma')
    ax_side.set_title("Key Factors for Placement")
    st.sidebar.pyplot(fig_side)

    with st.form("prediction_form"):
        st.subheader("Input Data Mahasiswa")
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "ME", "CE"])
        cgpa = st.number_input("CGPA", 0.0, 10.0, 3.5)
        tenth_percentage = st.number_input("10th %", 0.0, 100.0, 80.0)
        twelfth_percentage = st.number_input("12th %", 0.0, 100.0, 80.0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)

        study_hours = st.slider("Study Hours/Day", 1, 24, 6)
        sleep_hours = st.slider("Sleep Hours", 1, 12, 7)
        stress_level = st.slider("Stress Level", 1, 10, 2)
        attendance = st.slider("Attendance %", 0, 100, 85)

        coding_skill = st.slider("Coding Skill", 1, 5, 3)
        comm_skill = st.slider("Communication Skill", 1, 5, 3)
        aptitude_skill = st.slider("Aptitude Skill", 1, 5, 3)
        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Projects", 0, 10, 2)
        hackathons = st.number_input("Hackathons", 0, 10, 1)
        certs = st.number_input("Certifications", 0, 10, 1)

        extra_curr = st.selectbox("Extracurricular", ["Low", "Medium", "High"])
        part_time_job = st.selectbox("Part Time Job", ["No", "Yes"])
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        internet_access = st.selectbox("Internet Access", ["No", "Yes"])
        
        submitted = st.form_submit_button("Start Prediction")

    if submitted:
        input_data = {
            'gender': gender, 
            'branch': branch, 
            'cgpa': cgpa, 
            'tenth_percentage': tenth_percentage, 
            'twelfth_percentage': twelfth_percentage,
            'backlogs': backlogs, 
            'study_hours_per_day': study_hours, 
            'attendance_percentage': attendance, 
            'projects_completed': projects,
            'internships_completed': internships, 
            'coding_skill_rating': coding_skill,
            'communication_skill_rating': comm_skill, 
            'aptitude_skill_rating': aptitude_skill,
            'hackathons_participated': hackathons, 
            'certifications_count': certs,
            'sleep_hours': sleep_hours, 
            'stress_level': stress_level,
            'part_time_job': part_time_job, 
            'family_income_level': family_income,
            'city_tier': city_tier, 
            'internet_access': internet_access,
            'extracurricular_involvement': extra_curr
        }

        df = pd.DataFrame([input_data])
        df['academic_score'] = (df['cgpa'] * 0.5 + (df['tenth_percentage']/10) * 0.25 + (df['twelfth_percentage']/10) * 0.25)
        df['skill_score'] = df['coding_skill_rating'] + df['communication_skill_rating'] + df['aptitude_skill_rating']
        df['experience_score'] = (df['internships_completed'] * 2 + df['projects_completed'] + df['hackathons_participated'] + df['certifications_count'])

        final_features = [
            'gender', 'branch', 'cgpa', 'tenth_percentage', 'twelfth_percentage',
            'backlogs', 'study_hours_per_day', 'attendance_percentage',
            'projects_completed', 'internships_completed', 'coding_skill_rating',
            'communication_skill_rating', 'aptitude_skill_rating',
            'hackathons_participated', 'certifications_count', 'sleep_hours',
            'stress_level', 'part_time_job', 'family_income_level', 'city_tier',
            'internet_access', 'extracurricular_involvement', 'academic_score',
            'skill_score', 'experience_score'
        ]
        df_final = df[final_features]

        prediction_clf = clf_model.predict(df_final)[0]
        status = "PLACED" if prediction_clf == 1 else "NOT PLACED"
        prediction_reg = reg_model.predict(df_final)[0]

        st.divider()
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction_clf == 1:
                st.success(f"### Status: {status}")
                st.metric(label="Estimated Salary (LPA)", value=f"{prediction_reg:.2f}")
            else:
                st.error(f"### Status: {status}")
                st.warning("Keep improving your profile to increase chances!")

        with col_res2:
            st.write("**Your Scores vs Benchmark**")
            user_vals = [df['academic_score'].iloc[0], df['skill_score'].iloc[0], df['experience_score'].iloc[0]]
            benchmark_vals = [7.99, 11.15, 17.10] 
            labels = ['Academic', 'Skills', 'Experience']

            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            x_axis = np.arange(len(labels))
            ax_bar.bar(x_axis - 0.2, user_vals, 0.4, label='You', color='#4CAF50')
            ax_bar.bar(x_axis + 0.2, benchmark_vals, 0.4, label='Avg Placed', color='#FF9800', alpha=0.7)
            ax_bar.set_xticks(x_axis)
            ax_bar.set_xticklabels(labels)
            ax_bar.legend()
            st.pyplot(fig_bar)

        if prediction_clf == 1:
            st.write("Salary Benchmarking")
            avg_salary_dataset = 16.15 
            
            fig_salary, ax_salary = plt.subplots(figsize=(8, 3))
            salaries = [prediction_reg, avg_salary_dataset]
            names = ['Your Prediction', 'Market Average (Placed)']
            colors = ['#1f77b4', '#d62728']
            
            bars = ax_salary.barh(names, salaries, color=colors)
            ax_salary.set_xlabel('Salary in LPA')
            ax_salary.set_title('Perbandingan Estimasi Gaji Anda')
            ax_salary.bar_label(bars, fmt='%.2f', padding=5)
            ax_salary.set_xlim(0, max(salaries) * 1.3)
            
            st.pyplot(fig_salary)

if __name__ == "__main__":
    main()
