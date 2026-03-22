import os, base64, uuid, tempfile, math, io
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, send_file, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt
from PyPDF2 import PdfReader
from fpdf import FPDF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sympy import symbols, Eq, solve, sympify
import qrcode
import google.generativeai as genai

# ── Load env vars ──────────────────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")

# Fix Render postgres:// → postgresql://
db_url = os.environ.get("DATABASE_URL", "sqlite:///site.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ── Models ─────────────────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email    = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class UploadHistory(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    filename    = db.Column(db.String(200), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user        = db.relationship("User", backref=db.backref("uploads", lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup", methods=["GET"])
def signup_page():
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signup():
    username         = request.form["username"].strip()
    email            = request.form["email"].strip()
    password         = request.form["password"].strip()
    confirm_password = request.form["confirm-password"].strip()

    if not username or not email or not password or not confirm_password:
        flash("All fields are required.")
        return redirect(url_for("signup_page"))
    if password != confirm_password:
        flash("Passwords do not match!")
        return redirect(url_for("signup_page"))
    if User.query.filter_by(username=username).first():
        flash("Username already exists!")
        return redirect(url_for("signup_page"))

    hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    flash("Account created successfully! Please log in.")
    return redirect(url_for("login"))

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login_post():
    username = request.form["username"].strip()
    password = request.form["password"].strip()

    if not username or not password:
        flash("Both fields are required.")
        return redirect(url_for("login"))

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        session["username"] = username
        login_user(user)
        flash("Logged in successfully!")
        return redirect(url_for("dashboard"))
    else:
        flash("Invalid credentials")
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    logout_user()
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("home"))

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD & ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        flash("You must be logged in to view this page.")
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

@app.route("/account")
def my_account():
    if "username" not in session:
        flash("Please log in first.")
        return redirect(url_for("login"))
    user  = User.query.filter_by(username=session["username"]).first()
    email = user.email if user else "N/A"
    return render_template("account.html", username=session["username"], email=email)

@app.route("/upload_history")
@login_required
def upload_history():
    uploads = UploadHistory.query.filter_by(user_id=current_user.id).order_by(UploadHistory.upload_time.desc()).all()
    return render_template("history.html", uploads=uploads)

# ══════════════════════════════════════════════════════════════════════════════
# PDF CHAT  (uses Gemini — no torch/langchain)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/pdf_chat", methods=["GET", "POST"])
def pdf_chat():
    extracted_text = session.get("extracted_text", "")
    summary        = session.get("summary", "")
    answer         = session.get("answer", "")
    action         = request.form.get("action")
    question       = request.form.get("question")

    if request.method == "POST":
        file = request.files.get("pdf")

        if file and file.filename.endswith(".pdf"):
            session.pop("summary", None)
            session.pop("answer",  None)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                reader = PdfReader(tmp.name)
                extracted_text = "".join(page.extract_text() or "" for page in reader.pages)
            os.remove(tmp.name)
            session["extracted_text"] = extracted_text[:50000]

            if current_user.is_authenticated:
                db.session.add(UploadHistory(filename=file.filename, user_id=current_user.id))
                db.session.commit()

        if action == "summarize" and extracted_text:
            prompt   = f"Summarize the following document in clear bullet points:\n\n{extracted_text[:10000]}"
            response = gemini_model.generate_content(prompt)
            summary  = response.text
            session["summary"] = summary

        if action == "ask" and question and extracted_text:
            prompt = (
                f"Answer the question based ONLY on the document below.\n"
                f"Context:\n{extracted_text[:10000]}\n\n"
                f"Question: {question}"
            )
            response = gemini_model.generate_content(prompt)
            answer   = response.text
            session["answer"] = answer

    return render_template(
        "pdf_chat.html",
        extracted_text=session.get("extracted_text", ""),
        summary=session.get("summary", ""),
        answer=session.get("answer", ""),
        action=action,
    )

@app.route("/clear_pdf_session")
def clear_pdf_session():
    for key in ["extracted_text", "summary", "answer"]:
        session.pop(key, None)
    return redirect(url_for("pdf_chat"))

@app.route("/download_summary", methods=["POST"])
def download_summary():
    summary_text = request.form.get("summary_text", "")
    if not summary_text:
        return redirect(url_for("pdf_chat"))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, line)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        pdf_data = tmp.read()
    os.remove(tmp.name)

    return send_file(
        BytesIO(pdf_data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="summary.pdf",
    )

# ══════════════════════════════════════════════════════════════════════════════
# OCR EXTRACT  (uses Gemini Vision — no torch/TrOCR/layoutparser)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/ocr_extract", methods=["GET", "POST"])
def ocr_extract():
    raw_text          = ""
    handwritten_text  = ""
    structured_output = ""

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        filename = file.filename.lower()

        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_bytes = file.read()
            img_b64   = base64.b64encode(img_bytes).decode()
            ext  = filename.rsplit(".", 1)[-1]
            mime = f"image/{ext.replace('jpg','jpeg')}"

            raw_text = gemini_model.generate_content([
                {"mime_type": mime, "data": img_b64},
                "Extract ALL text from this image including printed and handwritten text. Return only the extracted text.",
            ]).text

            handwritten_text = gemini_model.generate_content([
                {"mime_type": mime, "data": img_b64},
                "Extract only handwritten text from this image. If none, say None.",
            ]).text

            structured_output = gemini_model.generate_content([
                {"mime_type": mime, "data": img_b64},
                "Describe the document structure: headings, tables, paragraphs, lists.",
            ]).text

        elif filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                reader = PdfReader(tmp.name)
                raw_text = "".join(p.extract_text() or "" for p in reader.pages)
            os.remove(tmp.name)
            structured_output = gemini_model.generate_content(
                f"Describe the structure and content of this PDF text:\n\n{raw_text[:5000]}"
            ).text

        elif filename.endswith(".csv"):
            df = pd.read_csv(file)
            raw_text = df.to_string()

        else:
            flash("Unsupported file format. Please upload PNG, JPG, PDF, or CSV.")
            return redirect(request.url)

    return render_template(
        "ocr_extract.html",
        raw_text=raw_text,
        structured_output=structured_output,
        handwritten_text=handwritten_text,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PLOTSPAN  (Excel → chart)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/plotspan_choice", methods=["GET"])
def plotspan_choice():
    return render_template("plotspan_choice.html")

@app.route("/plotspan", methods=["GET", "POST"])
def plotspan():
    plot_url   = None
    columns    = []
    excel_path = session.get("excel_path")

    if request.method == "POST":
        file = request.files.get("excel_file")
        if file and file.filename.endswith((".xls", ".xlsx")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                file.save(tmp.name)
                session["excel_path"] = tmp.name
                excel_path = tmp.name

        if excel_path and os.path.exists(excel_path):
            df      = pd.read_excel(excel_path)
            columns = df.columns.tolist()
            col1    = request.form.get("col1")
            col2    = request.form.get("col2")
            if col1 and col2 and col1 in columns and col2 in columns:
                fig, ax = plt.subplots()
                ax.plot(df[col1], df[col2])
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"{col1} vs {col2}")
                img = BytesIO()
                plt.tight_layout()
                fig.savefig(img, format="png")
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)

    return render_template("plotspan.html", plot_url=plot_url, columns=columns,
                           excel_path=excel_path)

# ── Step-based plotter ─────────────────────────────────────────────────────────

@app.route("/step1_upload", methods=["GET", "POST"])
def step1_upload():
    if request.method == "POST":
        file = request.files.get("excel_file")
        if file and file.filename.endswith((".xls", ".xlsx")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                file.save(tmp.name)
            temp_path = tmp.name
            try:
                df = pd.read_excel(temp_path)
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if not numeric_cols:
                    flash("No numeric columns found in the Excel file.")
                    os.unlink(temp_path)
                    return redirect(request.url)
                session["excel_path"] = temp_path
                return redirect(url_for("step2_chart"))
            except Exception as e:
                flash(f"Failed to read Excel file. {e}")
                os.unlink(temp_path)
                return redirect(request.url)
        else:
            flash("Please upload a valid Excel (.xls or .xlsx) file.")
    return render_template("plot_step1_upload.html")

@app.route("/step2", methods=["GET", "POST"])
def step2_chart():
    if "excel_path" not in session or not os.path.exists(session["excel_path"]):
        flash("Session expired or file missing.")
        return redirect(url_for("step1_upload"))

    if request.method == "POST":
        chart_type = request.form.get("chart_type")
        if chart_type in ["line", "bar", "pie"]:
            session["chart_type"] = chart_type
            return redirect(url_for("step3_select"))
        flash("Please select a valid chart type.")
    return render_template("plot_step2_chart.html")

@app.route("/step3", methods=["GET", "POST"])
def step3_select():
    if "excel_path" not in session or not os.path.exists(session["excel_path"]):
        flash("Session expired or file missing.")
        return redirect(url_for("step1_upload"))

    df         = pd.read_excel(session["excel_path"])
    chart_type = session.get("chart_type")
    plot_url   = None

    if request.method == "POST":
        x_axis = request.form.get("x_axis")
        y_axis = request.form.getlist("y_axis")

        try:
            fig, ax = plt.subplots()
            if chart_type == "pie":
                if len(y_axis) == 1:
                    data   = df[y_axis[0]].dropna()
                    labels = df[x_axis].dropna()
                    ax.pie(data, labels=labels, autopct="%1.1f%%")
                else:
                    flash("Select exactly one column for a pie chart.")
                    return redirect(request.url)
            elif chart_type == "line":
                for col in y_axis:
                    ax.plot(df[x_axis], df[col], label=col)
                ax.set_xlabel(x_axis)
                ax.legend()
            elif chart_type == "bar":
                x     = np.arange(len(df[x_axis]))
                bar_w = 0.8 / len(y_axis)
                for j, col in enumerate(y_axis):
                    ax.bar(x + j * bar_w, df[col], width=bar_w, label=col)
                ax.set_xticks(x + bar_w * (len(y_axis) - 1) / 2)
                ax.set_xticklabels(df[x_axis])
                ax.legend()
            else:
                flash("Unsupported chart type.")
                return redirect(request.url)

            img = BytesIO()
            plt.tight_layout()
            fig.savefig(img, format="png")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            session["chart_image_path_b64"] = plot_url

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp_file.name, format="png")
            session["chart_image_path"] = tmp_file.name
            plt.close(fig)

        except Exception as e:
            flash(f"Error generating chart: {str(e)}")
            return redirect(request.url)

    return render_template("plot_step3_select.html",
                           columns=df.columns.tolist(),
                           chart_type=chart_type,
                           plot_url=plot_url)

@app.route("/download-chart")
def download_chart():
    chart_path = session.get("chart_image_path")
    if chart_path and os.path.exists(chart_path):
        return send_file(chart_path, as_attachment=True)
    flash("Chart image not available.")
    return redirect(url_for("step3_select"))

# ══════════════════════════════════════════════════════════════════════════════
# EQUATION PLOTTER & SOLVER
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/plot_equation", methods=["GET", "POST"])
def plot_equation():
    plot_url = None

    if request.method == "POST":
        equations = request.form.get("equation")
        x_start   = float(request.form.get("x_start"))
        x_end     = float(request.form.get("x_end"))
        y_min     = float(request.form.get("y_min"))
        y_max     = float(request.form.get("y_max"))

        try:
            fig, ax = plt.subplots()
            x = np.linspace(x_start, x_end, 500)
            local_dict = {
                "x": x, "np": np,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "pi": np.pi, "e": np.e, "abs": np.abs,
            }
            for eq in equations.split(";"):
                eq = eq.strip()
                if not eq:
                    continue
                y = eval(eq, {"__builtins__": {}}, local_dict)
                ax.plot(x, y, label=eq)

            ax.set_xlabel("x-axis")
            ax.set_ylabel("y-axis")
            ax.set_title("Plot of Equation(s)")
            ax.grid(True)
            ax.legend()
            ax.set_xlim(x_start, x_end)
            ax.set_ylim(y_min, y_max)

            img = BytesIO()
            plt.tight_layout()
            fig.savefig(img, format="png")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp_file.name, format="png")
            session["equation_plot_path"] = tmp_file.name
            plt.close(fig)

        except Exception as e:
            flash(f"Error plotting equation(s): {str(e)}")
            return redirect(url_for("plot_equation"))

    return render_template("plot_equation.html", plot_url=plot_url)

@app.route("/download_equation_plot")
def download_equation_plot():
    path = session.get("equation_plot_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("Plot not found.")
    return redirect(url_for("plot_equation"))

@app.route("/solve_equation", methods=["POST"])
def solve_equation():
    eq_input = request.form.get("solve_equation")
    try:
        raw_eqs = [e.strip() for e in eq_input.split(";") if e.strip()]
        if len(raw_eqs) == 1 and "=" not in raw_eqs[0]:
            raw_eqs[0] = raw_eqs[0] + "=0"

        equations   = [Eq(*map(sympify, eq.split("="))) for eq in raw_eqs if "=" in eq]
        all_symbols = sorted(set().union(*[eq.free_symbols for eq in equations]), key=lambda s: s.name)

        if not equations:
            result = "⚠ No valid equations were provided."
        else:
            sol = solve(equations, *all_symbols, dict=True)
            if sol:
                result = f"✅ Solution(s): {sol}"
            else:
                single_expr  = sympify(eq_input.strip())
                vars_in_expr = list(single_expr.free_symbols)
                if vars_in_expr:
                    sol    = solve(single_expr, vars_in_expr[0])
                    result = f"✅ Solution(s): {sol}"
                else:
                    result = "⚠ Could not extract variables to solve."
    except Exception as e:
        result = f"❌ Error solving equation(s): {str(e)}"

    return render_template("solve_result.html", result=result)

# ══════════════════════════════════════════════════════════════════════════════
# QR CODE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/qr", methods=["GET", "POST"])
def text_to_qr():
    qr_image    = None
    qr_filename = None

    if request.method == "POST":
        text = request.form.get("text")
        qr   = qrcode.make(text)

        filename  = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join("static", "qr_codes")
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        qr.save(full_path)

        img_io = BytesIO()
        qr.save(img_io, "PNG")
        img_io.seek(0)
        qr_image    = base64.b64encode(img_io.getvalue()).decode()
        qr_filename = filename

    return render_template("qr_generator.html", qr_image=qr_image, qr_filename=qr_filename)

@app.route("/download/<filename>")
def download_qr(filename):
    file_path = os.path.join("static", "qr_codes", filename)
    return send_file(file_path, as_attachment=True)

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
