from flask import Flask, jsonify, render_template, request

from llm_module import generate_feedback
from model import analyze_resume
from utils import extract_text, require_text, save_upload


UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def error_response(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "AI Resume Analyzer"})


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/website/analyze")
def website_analyze():
    try:
        resume_text = request.form.get("resume_text", "")
        uploaded_file = request.files.get("resume")
        if uploaded_file and uploaded_file.filename:
            path = save_upload(uploaded_file, app.config["UPLOAD_FOLDER"])
            resume_text = extract_text(path)

        resume_text = require_text(resume_text, "resume text or resume file")
        job_description = request.form.get("job_description", "")
        analysis = analyze_resume(resume_text, job_description).to_dict()
        llm_feedback = generate_feedback(resume_text, job_description, analysis)

        return render_template(
            "index.html",
            analysis=analysis,
            feedback=llm_feedback.get("feedback"),
            provider=llm_feedback.get("provider"),
            warning=llm_feedback.get("warning"),
            job_description=job_description,
        )
    except Exception as exc:
        return render_template("index.html", error=str(exc)), 400


@app.post("/upload_resume")
def upload_resume():
    try:
        file = request.files.get("resume")
        path = save_upload(file, app.config["UPLOAD_FOLDER"])
        text = extract_text(path)
        return jsonify({"filename": file.filename, "text": text, "characters": len(text)})
    except Exception as exc:
        return error_response(str(exc))


@app.post("/analyze")
def analyze():
    try:
        resume_text = request.form.get("resume_text", "")
        if "resume" in request.files:
            path = save_upload(request.files["resume"], app.config["UPLOAD_FOLDER"])
            resume_text = extract_text(path)

        resume_text = require_text(resume_text, "resume_text or resume file")
        job_description = request.form.get("job_description", "")
        result = analyze_resume(resume_text, job_description)
        return jsonify(result.to_dict())
    except Exception as exc:
        return error_response(str(exc))


@app.post("/generate_feedback")
def feedback():
    try:
        resume_text = request.form.get("resume_text", "")
        if "resume" in request.files:
            path = save_upload(request.files["resume"], app.config["UPLOAD_FOLDER"])
            resume_text = extract_text(path)

        resume_text = require_text(resume_text, "resume_text or resume file")
        job_description = request.form.get("job_description", "")
        analysis = analyze_resume(resume_text, job_description).to_dict()
        llm_feedback = generate_feedback(resume_text, job_description, analysis)

        return jsonify({"analysis": analysis, **llm_feedback})
    except Exception as exc:
        return error_response(str(exc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
