import requests
import streamlit as st


API_URL = st.sidebar.text_input("API URL", "http://localhost:5000")

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")

resume = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Job description", height=220)

if st.button("Analyze Resume", type="primary"):
    if not resume:
        st.error("Upload a resume first.")
    else:
        files = {"resume": (resume.name, resume.getvalue())}
        data = {"job_description": job_description}
        response = requests.post(f"{API_URL}/generate_feedback", files=files, data=data, timeout=90)

        if response.ok:
            payload = response.json()
            analysis = payload.get("analysis", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Role", analysis.get("predicted_role", "Unknown"))
            col2.metric("Match Score", f"{analysis.get('score', 0)}%")
            col3.metric("ATS Score", f"{analysis.get('ats_score', 0)}/100")

            st.subheader("Detected Skills")
            st.write(", ".join(analysis.get("skills", [])) or "No skills detected.")

            st.subheader("LLM Feedback")
            st.markdown(payload.get("feedback", "No feedback generated."))
            st.caption(f"Provider: {payload.get('provider')}")
        else:
            st.error(response.text)
