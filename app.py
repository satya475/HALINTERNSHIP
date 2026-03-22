from flask import Flask, render_template, request, redirect, send_file, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt
from PyPDF2 import PdfReader
import tempfile, uuid, os, base64, math, io
from io import BytesIO
from datetime import datetime
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, Eq, solve, sympify
import qrcode
from fpdf import FPDF
from dotenv import load_dotenv

# ── Gemini setup ─────────────────────────────────
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── Flask app setup (same as before) ─────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    
    # Apply autocontrast
    image = ImageOps.autocontrast(image)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    return image

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
            session.pop('summary', None)
            session.pop('answer', None)

            # ── Extract text from PDF using PyPDF2 (no torch needed) ──
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                reader = PdfReader(tmp.name)
                extracted_text = "".join(
                    page.extract_text() or "" for page in reader.pages
                )
            os.remove(tmp.name)
            session['extracted_text'] = extracted_text[:50000]  # Gemini has large context

            # Save upload history (same as before)
            new_upload = UploadHistory(filename=file.filename, user_id=current_user.id)
            db.session.add(new_upload)
            db.session.commit()

        if action == "summarize" and extracted_text:
            # ── Use Gemini to summarize (replaces langchain summarizer) ──
            prompt = f"Summarize the following document in clear bullet points:\n\n{extracted_text[:10000]}"
            response = gemini_model.generate_content(prompt)
            summary = response.text
            session['summary'] = summary

        if action == "ask" and question and extracted_text:
            # ── Use Gemini to answer questions (replaces qa_pipeline + faiss) ──
            prompt = f"""Answer the question based ONLY on the document below.
Context:
{extracted_text[:10000]}

Question: {question}"""
            response = gemini_model.generate_content(prompt)
            answer = response.text
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                file.save(tmp.name)
            temp_path = tmp.name  # store the path
            try:
                df = pd.read_excel(temp_path)

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


##

@app.route('/solve_equation', methods=['POST'])
def solve_equation():
    eq_input = request.form.get('solve_equation')

    try:
        # Split by semicolon to support multiple equations
        raw_eqs = [e.strip() for e in eq_input.split(';') if e.strip()]

        # If there's only one equation and no '=', treat as expression == 0
        if len(raw_eqs) == 1 and '=' not in raw_eqs[0]:
            raw_eqs[0] = raw_eqs[0] + '=0'

        # Create list of Eq(...) objects
        equations = [Eq(*map(sympify, eq.split('='))) for eq in raw_eqs if '=' in eq]

        # Extract all unique symbols
        all_symbols = sorted(set().union(*[eq.free_symbols for eq in equations]), key=lambda s: s.name)

        if not equations:
            result = "⚠️ No valid equations were provided."
        else:
            # Attempt to solve symbolically
            sol = solve(equations, *all_symbols, dict=True)

            if sol:
                result = f"✅ Solution(s): {sol}"
            else:
                # Try solving as an expression (e.g., a single nonlinear equation)
                single_expr = sympify(eq_input.strip())
                vars_in_expr = list(single_expr.free_symbols)
                if vars_in_expr:
                    sol = solve(single_expr, vars_in_expr[0])
                    result = f"✅ Solution(s): {sol}"
                else:
                    result = "⚠️ Could not extract variables to solve."

    except Exception as e:
        result = f"❌ Error solving equation(s): {str(e)}"

    return render_template('solve_result.html', result=result)


##

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

        # ── IMAGE: send directly to Gemini Vision ──
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_bytes = file.read()
            img_b64 = base64.b64encode(img_bytes).decode()
            ext = filename.rsplit('.', 1)[-1]
            mime = f"image/{ext.replace('jpg','jpeg')}"

            response = gemini_model.generate_content([
                {"mime_type": mime, "data": img_b64},
                "Extract ALL text from this image. Include printed text and handwriting. Return only the extracted text."
            ])
            raw_text = response.text

            response2 = gemini_model.generate_content([
                {"mime_type": mime, "data": img_b64},
                "Extract any handwritten text from this image only."
            ])
            handwritten_text = response2.text

            response3 = gemini_model.generate_content([
                {"mime_type": mime, "data": img_b64},
                "Describe the structure of this document: headings, tables, paragraphs, lists."
            ])
            structured_output = response3.text

        # ── PDF: extract text, then use Gemini for structure ──
        elif filename.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                reader = PdfReader(tmp.name)
                raw_text = "".join(p.extract_text() or "" for p in reader.pages)
            os.remove(tmp.name)

            response = gemini_model.generate_content(
                f"Describe the structure and content of this PDF text:\n\n{raw_text[:5000]}"
            )
            structured_output = response.text

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

# @app.route('/qr', methods=['GET', 'POST'])
# def text_to_qr():
#     qr_image_base64 = None
#     if request.method == 'POST':
#         text = request.form['text']
#         qr = qrcode.make(text)
#         img_io = io.BytesIO()
#         qr.save(img_io, 'PNG')
#         img_io.seek(0)
#         qr_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
#     return render_template('qr_generator.html', qr_image=qr_image_base64)


@app.route('/qr', methods=['GET', 'POST'])
def text_to_qr():
    qr_image_base64 = None
    filename = None
    if request.method == 'POST':
        text = request.form['text']
        qr = qrcode.make(text)

        # Save image to server
        filename = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join('static', 'qr_codes')
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        qr.save(full_path)

        # Convert to base64 for display
        img_io = io.BytesIO()
        qr.save(img_io, 'PNG')
        img_io.seek(0)
        qr_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return render_template('qr_generator.html', qr_image=qr_image_base64, qr_filename=filename)

@app.route('/download/<filename>')
def download_qr(filename):
    file_path = os.path.join('static', 'qr_codes', filename)
    return send_file(file_path, as_attachment=True)
##

@app.route('/account')
def my_account():
    if 'username' not in session:
        flash('Please log in first.')
        return redirect(url_for('login_page'))

    user = User.query.filter_by(username=session['username']).first()
    email = user.email if user else 'N/A'

    return render_template('account.html', username=user.username, email=email)

@app.route('/upload_history')
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


