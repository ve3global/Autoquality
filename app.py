from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RULES = [
    {'id': 'missing_values', 'name': 'Check Missing Values', 'description': 'Identify missing values'},
    {'id': 'whitespace', 'name': 'Check Whitespace', 'description': 'Detect extra spaces'},
    {'id': 'outliers', 'name': 'Check Outliers', 'description': 'Find statistical outliers'},
    {'id': 'format', 'name': 'Check Format', 'description': 'Validate email, date, and contact formats'},
    {'id': 'duplicates', 'name': 'Check Duplicates', 'description': 'Find duplicate records'},
    {'id': 'data_inconsistency', 'name': 'Check Data Inconsistency', 'description': 'Detect inconsistent formatting'}
]

def check_missing_values(df):
    return list(df[df.isnull().any(axis=1)].index)

def check_whitespace(df):
    return list(df[df.applymap(lambda x: isinstance(x, str) and (x.startswith(' ') or x.endswith(' ') or '  ' in x))].index)

def check_outliers(df):
    flagged_indices = []
    for col in df.select_dtypes(include=['int64', 'float64']):
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        flagged_indices.extend(df[z_scores > 3].index.tolist())
    return list(set(flagged_indices))

def check_format(df):
    flagged_indices = []
    if 'email' in df.columns:
        flagged_indices.extend(df[~df['email'].astype(str).str.match(r'^[^@]+@[^@]+\\.[^@]+$', na=False)].index.tolist())
    if 'contact' in df.columns:
        flagged_indices.extend(df[~df['contact'].astype(str).str.match(r'^[0-9]{10}$', na=False)].index.tolist())
    return list(set(flagged_indices))

def check_duplicates(df):
    return list(df[df.duplicated()].index)

def check_data_inconsistency(df):
    flagged_indices = set()
    for col in df.columns:
        if df[col].dtype == object:
            inconsistent_mask = df[col].astype(str).str.match(r'[^a-zA-Z0-9\s]', na=False)
            flagged_indices.update(df[inconsistent_mask].index.tolist())
    return list(flagged_indices)

@app.route('/', methods=['GET', 'POST'])  # Allow both GET & POST
def upload_file():
    if request.method == 'POST':  
        if 'file' not in request.files:
            return "No file part", 400  # Handle missing file

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400  # Handle empty file

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

        return redirect(url_for('upload_file'))  # Refresh page after upload

    # Fetch uploaded files for display
    files = [{'name': f, 'date': datetime.fromtimestamp(os.path.getmtime(os.path.join(UPLOAD_FOLDER, f))).strftime('%Y-%m-%d %H:%M:%S')}
             for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]

    return render_template('/templates/saved_data.html', files=files)

@app.route('/autoquality/<file_name>')
def autoquality(file_name):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    session['uploaded_file'] = file_path

    df = pd.read_csv(file_path)
    print("Dataframe:", df)
    return render_template('/templates/autoquality.html', 
                           tables=[df.to_html(classes='table', table_id='dataTable')], 
                           rules=RULES, file_name=file_name)

@app.route('/run_rule/<file_name>/<rule_id>')
def run_rule(file_name, rule_id):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    df = pd.read_csv(file_path)

    rule_functions = {
        'missing_values': check_missing_values,
        'whitespace': check_whitespace,
        'outliers': check_outliers,
        'format': check_format,
        'duplicates': check_duplicates,
        'data_inconsistency': check_data_inconsistency
    }

    flagged_indices = rule_functions[rule_id](df) if rule_id in rule_functions else []
    quality_score = max(0, (len(df) - len(flagged_indices)) / len(df) * 100)

    return jsonify({'flagged_indices': flagged_indices, 'quality_score': quality_score})

if __name__ == '__main__':
    app.run(debug=True)
