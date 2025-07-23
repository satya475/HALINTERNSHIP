from flask import Flask, render_template, request, redirect, send_file, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt
from PyPDF2 import PdfReader
from transformers import DonutProcessor, pipeline, VisionEncoderDecoderModel, TrOCRProcessor
import tempfile
import os
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
from io import BytesIO
import base64
from datetime import datetime
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r"C:\Users\HAL\Downloads"
from pdf2image import convert_from_path  
poppler_path = r"C:\Users\HAL\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# images = convert_from_path("your_file.pdf", poppler_path=poppler_path)
import torch
import layoutparser as lp
import cv2
import numpy as np
from sympy import symbols, Eq, solve, sympify
import math
import qrcode


def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    
    # Apply autocontrast
    image = ImageOps.autocontrast(image)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    return image



app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database & Auth Setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Transformers
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

#Donut model and processor 
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# TrOCR for handwritten text
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)


# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username'].strip()
    email = request.form['email'].strip()
    password = request.form['password'].strip()
    confirm_password = request.form['confirm-password'].strip()

    if not username or not email or not password or not confirm_password:
        flash('All fields are required.')
        return redirect(url_for('signup_page'))

    if password != confirm_password:
        flash('Passwords do not match!')
        return redirect(url_for('signup_page'))

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username already exists!')
        return redirect(url_for('signup_page'))

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()



    flash('Account created successfully! Please log in.')
    return redirect(url_for('login_page'))


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username'].strip()
    password = request.form['password'].strip()

    if not username or not password:
        flash('Both fields are required.')
        return redirect(url_for('login_page'))

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        session['username'] = username
        login_user(user)  # ✅ This sets current_user
        flash('Logged in successfully!')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials')
        return redirect(url_for('login_page'))



@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('You must be logged in to view this page.')
        return redirect(url_for('login_page'))

    return render_template('dashboard.html', username=session['username'])


@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    extracted_text = session.get('extracted_text', "")
    summary = session.get('summary', "")
    answer = ""
    action = request.form.get("action")
    question = request.form.get("question")

    if request.method == 'POST':
        file = request.files.get('pdf')

        if file and file.filename.endswith('.pdf'):
            # Clear old results
            session.pop('summary', None)
            session.pop('answer', None)

            # Save temporarily and extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                temp_path = tmp.name

            reader = PdfReader(temp_path)
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
            reader.stream.close()

            os.remove(temp_path)

            session['extracted_text'] = extracted_text

            # Save upload history
            new_upload = UploadHistory(filename=file.filename, user_id=current_user.id)
            db.session.add(new_upload)
            db.session.commit()


        if action == "summarize" and extracted_text:
            max_chunk = 2000
            chunks = [extracted_text[i:i + max_chunk] for i in range(0, len(extracted_text), max_chunk)]
            summary = ""
            for c in chunks:
                result = summarizer(c, max_length=200, min_length=50, do_sample=False)
                summary += result[0]['summary_text'] + " "
            summary = summary.strip()
            session['summary'] = summary

        if action == "ask" and question and extracted_text:
            prompt = f"Answer the question based on the context.\nContext: {extracted_text[:3000]}\nQuestion: {question}"
            result = qa_pipeline(prompt, max_length=128, do_sample=False)
            answer = result[0]['generated_text']
            session['answer'] = answer

    return render_template(
        'pdf_chat.html',
        extracted_text=session.get('extracted_text', ""),
        summary=session.get('summary', ""),
        answer=session.get('answer', ""),
        action=action
    )

@app.route('/download_summary', methods=['POST'])
def download_summary():
    summary_text = request.form.get('summary_text', '')
    if not summary_text:
        return redirect(url_for('pdf_chat'))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 10, line)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        pdf_data = tmp.read()

    os.remove(tmp.name)

    return send_file(
        io.BytesIO(pdf_data),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='summary.pdf'
    )

@app.route('/clear_pdf_session')
def clear_pdf_session():
    for key in ['extracted_text', 'summary', 'answer']:
        session.pop(key, None)
    return redirect(url_for('pdf_chat'))




@app.route('/plotspan_choice', methods=['GET'])
def plotspan_choice():
    return render_template('plotspan_choice.html')

@app.route('/plot_equation', methods=['GET', 'POST'])
def plot_equation():
    plot_url = None
    solution_output = None

    if request.method == 'POST':
        equations = request.form.get('equation')
        x_start = float(request.form.get('x_start'))
        x_end = float(request.form.get('x_end'))
        y_min = float(request.form.get('y_min'))
        y_max = float(request.form.get('y_max'))

        try:
            fig, ax = plt.subplots()
            x = np.linspace(x_start, x_end, 500)
            local_dict = {"x": x, "np": np, "sin": np.sin, "cos": np.cos,
                          "tan": np.tan, "exp": np.exp, "log": np.log,
                          "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "abs": np.abs}

            for eq in equations.split(';'):
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
            fig.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)

            # Save for download
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            with open(tmp_file.name, 'wb') as f:
                f.write(img.getbuffer())
            session['equation_plot_path'] = tmp_file.name

        except Exception as e:
            flash(f"Error plotting equation(s): {str(e)}")
            return redirect(url_for('plot_equation'))

    return render_template('plot_equation.html', plot_url=plot_url)


@app.route('/solve_equation', methods=['POST'])
def solve_equation():
    eq_input = request.form.get('solve_equation')
    try:
        # Parse equations (e.g., "x + y = 2; x - y = 0")
        eqs = [Eq(*map(sympify, e.split('='))) for e in eq_input.split(';') if '=' in e]
        all_symbols = list(set().union(*[e.free_symbols for e in eqs]))
        sol = solve(eqs, *all_symbols, dict=True)

        if not sol:
            result = "❌ No solution or complex solution found."
        else:
            result = f"✅ Solution(s): {sol}"

    except Exception as e:
        result = f"⚠️ Error solving equations: {str(e)}"

    return render_template('solve_result.html', result=result)

@app.route('/download_equation_plot')
def download_equation_plot():
    path = session.get('equation_plot_path')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("Plot not found.")
    return redirect(url_for('plot_equation'))



@app.route('/step1_upload', methods=['GET', 'POST'])
def step1_upload():
    if request.method == 'POST':
        file = request.files.get('excel_file')
        if file and file.filename.endswith(('.xls', '.xlsx')):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            file.save(tmp.name)
            try:
                df = pd.read_excel(tmp.name)
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                if not numeric_cols:
                    flash("No numeric columns found in the Excel file.")
                    os.unlink(tmp.name)
                    return redirect(request.url)
                session['excel_path'] = tmp.name
                return redirect(url_for('step2_chart'))
            except Exception as e:
                flash("Failed to read Excel file.")
                os.unlink(tmp.name)
                return redirect(request.url)
        else:
            flash("Please upload a valid Excel (.xls or .xlsx) file.")
    return render_template('plot_step1_upload.html')


@app.route('/step2', methods=['GET', 'POST'])
def step2_chart():
    if 'excel_path' not in session or not os.path.exists(session['excel_path']):
        flash("Session expired or file missing.")
        return redirect(url_for('step1_upload'))

    if request.method == 'POST':
        chart_type = request.form.get('chart_type')
        if chart_type in ['line', 'bar', 'pie']:
            session['chart_type'] = chart_type
            return redirect(url_for('step3_select'))
        flash("Please select a valid chart type.")
    return render_template('plot_step2_chart.html')


@app.route('/step3', methods=['GET', 'POST'])
def step3_select():
    # Step 0: Check if Excel file exists
    if 'excel_path' not in session or not os.path.exists(session['excel_path']):
        flash("Session expired or file missing.")
        return redirect(url_for('step1_upload'))

    df = pd.read_excel(session['excel_path'])
    chart_type = session.get('chart_type')
    plot_url = None

    if request.method == 'POST':
        x_axis = request.form.get('x_axis')
        y_axis = request.form.getlist('y_axis')  # Handles multiple checkbox values

        try:
            fig, ax = plt.subplots()

            if chart_type == 'pie':
                if len(y_axis) == 1:
                    data = df[y_axis[0]].dropna()
                    labels = df[x_axis].dropna()
                    ax.pie(data, labels=labels, autopct='%1.1f%%')
                else:
                    flash("Select exactly one column for a pie chart.")
                    return redirect(request.url)

            elif chart_type == 'line':
                for col in y_axis:
                    ax.plot(df[x_axis], df[col], label=col)
                ax.set_xlabel(x_axis)
                ax.legend()

            elif chart_type == 'bar':
                x = np.arange(len(df[x_axis]))
                bar_width = 0.8 / len(y_axis)
                for i, col in enumerate(y_axis):
                    ax.bar(x + i * bar_width, df[col], width=bar_width, label=col)
                ax.set_xlabel(x_axis)
                ax.set_xticks(x + bar_width * (len(y_axis) - 1) / 2)
                ax.set_xticklabels(df[x_axis])
                ax.legend()

            else:
                flash("Unsupported chart type.")
                return redirect(request.url)

            # Save to base64 for preview
            img = BytesIO()
            plt.tight_layout()
            fig.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)

            # Save to temp file for download
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            with open(tmp_file.name, 'wb') as f:
                f.write(img.getbuffer())
            session['chart_image_path'] = tmp_file.name

        except Exception as e:
            flash(f"Error generating chart: {str(e)}")
            return redirect(request.url)

    return render_template(
        'plot_step3_select.html',
        columns=df.columns,
        chart_type=chart_type,
        plot_url=plot_url
    )

@app.route('/download-chart')
def download_chart():
    chart_path = session.get('chart_image_path')
    if chart_path and os.path.exists(chart_path):
        return send_file(chart_path, as_attachment=True)
    flash("Chart image not available.")
    return redirect(url_for('step3_select'))





@app.route('/ocr_extract', methods=['GET', 'POST'])
def ocr_extract():
    raw_text = ""
    handwritten_text = ""
    structured_output = ""

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        filename = file.filename.lower()

        # ─── IMAGE HANDLING ───────────────────────────
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file.stream).convert("RGB")
            preprocessed_img = preprocess_image(image)

            # (1) pytesseract (layout-preserved OCR)
            raw_text = pytesseract.image_to_string(preprocessed_img)

            # (2) TrOCR (handwritten text recognition)
            try:
                trocr_pixel = trocr_processor(images=preprocessed_img, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    trocr_output_ids = trocr_model.generate(trocr_pixel)
                handwritten_text = trocr_processor.batch_decode(trocr_output_ids, skip_special_tokens=True)[0]
            except Exception as e:
                handwritten_text = f"[ERROR running TrOCR]: {e}"

            # (3) Donut (field extraction)
            try:
                donut_pixel = processor(preprocessed_img, return_tensors="pt").pixel_values.to(device)
                task_prompt = "<s_docvqa><s_question>What does the document contain?</s_question><s_answer>"
                decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        donut_pixel,
                        decoder_input_ids=decoder_input_ids,
                        max_length=512,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True
                    )
                structured_output = processor.batch_decode(outputs.sequences)[0]
                structured_output = structured_output.replace(processor.tokenizer.eos_token, "").replace("<pad>", "").strip()
            except Exception as e:
                structured_output = f"[ERROR running Donut]: {e}"

        # ─── PDF HANDLING ─────────────────────────────
        elif filename.endswith('.pdf'):
            import tempfile
            from PyPDF2 import PdfReader
            from pdf2image import convert_from_path

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    file.save(tmp.name)
                    pdf_path = tmp.name

                # Extract text directly from PDF
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    raw_text += page.extract_text() or ""

                # Also OCR any images in the PDF (optional)
                images = convert_from_path(pdf_path)
                for img in images:
                    pre_img = preprocess_image(img)
                    raw_text += "\n" + pytesseract.image_to_string(pre_img)

                os.remove(pdf_path)
            except Exception as e:
                raw_text = f"[ERROR reading PDF]: {e}"

        # ─── CSV HANDLING ─────────────────────────────
        elif filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                raw_text = df.to_string(index=False)
            except Exception as e:
                raw_text = f"[ERROR reading CSV]: {e}"

        else:
            flash("Unsupported file format.")
            return redirect(request.url)

    return render_template(
        'ocr_extract.html',
        raw_text=raw_text,
        structured_output=structured_output,
        handwritten_text=handwritten_text
    )


##Add new code here

@app.route('/qr', methods=['GET', 'POST'])
def text_to_qr():
    qr_image_base64 = None
    if request.method == 'POST':
        text = request.form['text']
        qr = qrcode.make(text)
        img_io = io.BytesIO()
        qr.save(img_io, 'PNG')
        img_io.seek(0)
        qr_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return render_template('qr_generator.html', qr_image=qr_image_base64)

##

@app.route('/account')
def my_account():
    if 'username' not in session:
        flash('Please log in first.')
        return redirect(url_for('login_page'))

    user = User.query.filter_by(username=session['username']).first()
    email = user.email if user else 'N/A'

    return render_template('account.html', username=user.username, email=email)

@app.route('/history')
@login_required
def upload_history(): 
    uploads = UploadHistory.query.filter_by(user_id=current_user.id).order_by(UploadHistory.upload_time.desc()).all()
    return render_template('history.html', uploads=uploads)

@app.route('/logout')
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('home'))


class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('uploads', lazy=True))
     
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


