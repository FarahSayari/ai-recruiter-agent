# app.py
import streamlit as st
import requests

st.set_page_config(page_title="AI Recruiter Agent", page_icon="🤖", layout="centered")

st.title("🤖 AI Recruiter Agent")
st.write("Enter a job description and let the AI match top candidates!")

job_desc = st.text_area("Job Description", height=200)
if st.button("Run Agent"):
    if job_desc.strip():
        with st.spinner("Running AI agent..."):
            payload = {"job_description": job_desc, "top_k": 5, "with_explain": True}
            res = requests.post("http://localhost:8000/query", json=payload)

        if res.status_code == 200:
            data = res.json()["results"]
            for r in data:
                with st.chat_message("assistant"):
                    st.markdown(f"**🧾 Candidate ID:** {r['candidate_id']} | **Score:** {r['score']:.4f}")
                    st.markdown(r.get("explanation", "_No explanation available._"))
        else:
            st.error(f"Error {res.status_code}: {res.text}")
    else:
        st.warning("Please enter a job description first.")
