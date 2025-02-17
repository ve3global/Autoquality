from flask import Flask, render_template, request, redirect, send_file, url_for, session, jsonify
import pandas as pd
import os
import numpy as np
from datetime import datetime
from pandas.api.types import infer_dtype

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the rules
RULES = [
    {'id': 'empty_rows', 'name': 'Check Empty Rows', 'description': 'Detect empty rows'},
    {'id': 'duplicates', 'name': 'Check Duplicates', 'description': 'Find duplicate records'},
    {'id': 'missing_values', 'name': 'Check Missing Values', 'description': 'Identify missing values'},
    {'id': 'whitespace', 'name': 'Check Whitespace', 'description': 'Detect extra spaces'},
    {'id': 'email_format', 'name': 'Check Email Format', 'description': 'Validate email format'},
    {'id': 'data_types', 'name': 'Check Data Types', 'description': 'Ensure columns have correct data types'},
    {'id': 'categorical_values', 'name': 'Check Categorical Values', 'description': 'Ensure categorical data matches predefined categories'},
    {'id': 'negative_values', 'name': 'Check Negative Values', 'description': 'Detect negative values'},
    {'id': 'outliers', 'name': 'Check Outliers', 'description': 'Find statistical outliers'},
    {'id': 'special_char', 'name': 'Check Special Characters', 'description': 'Check for Special Char in ID'},
    # {'id': 'boolean_values', 'name': 'Check Boolean Values', 'description': 'Validate boolean values'},
    # {'id': 'contact_format', 'name': 'Check Contact Format', 'description': 'Validate contact format'},
    # {'id': 'data_inconsistency', 'name': 'Check Data Inconsistency', 'description': 'Detect inconsistent formatting'},
    # {'id': 'ranges', 'name': 'Check Ranges', 'description': 'Validate numeric values within acceptable ranges'},
    # {'id': 'cross_field_validation', 'name': 'Cross-Field Validation', 'description': 'Ensure logical consistency between related fields'},
    # {'id': 'normalize_data', 'name': 'Normalize Data', 'description': 'Scale numeric data to a standard range'},
    # {'id': 'encode_categorical_data', 'name': 'Encode Categorical Data', 'description': 'Convert categorical data into numerical formats'},
    # {'id': 'anonymize_data', 'name': 'Anonymize Sensitive Data', 'description': 'Remove or encrypt personally identifiable information (PII)'}
]

# Rule Functions
def scan_empty_rows(df):
    return df[df.isnull().all(axis=1)].index.tolist()

def fix_empty_rows(df):
    return df.dropna(how='all').reset_index(drop=True)


def scan_duplicates(df):
    # print("Checking duplicates...", list(df[df.duplicated()].index))
    return list(df[df.duplicated()].index)
    
def fix_duplicates(df):
    # return df.iloc[:1].append(df.iloc[1:].drop_duplicates(), ignore_index=True)
    return df.drop_duplicates(keep='first')
    # df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # return df.drop_duplicates()
    # df = df.astype(str)  # Convert all columns to string format
    # return df.drop_duplicates()


def scan_missing_values(df):
    # print("Checking missing values...",list(df[df.isnull().any(axis=1)].index))
    return list(df[df.isnull().any(axis=1)].index)

def fix_missing_values(df):
    # print("Fixing missing values...")
    # df.fillna(method='ffill', inplace=True)
    return df.fillna("NA")


def scan_whitespace(df):
    #print("Checking whitespace...", list(df[df.apply(lambda col: col.map(lambda x: isinstance(x, str) and (x.startswith(' ') or x.endswith(' ') or '  ' in x)), axis=0)].index))
    return list(df[df.apply(lambda col: col.map(lambda x: isinstance(x, str) and (x.startswith(' ') or x.endswith(' ') or '  ' in x)), axis=0)].index)

def fix_whitespace(df):
    return df.apply(lambda col: col.map(lambda x: " ".join(str(x).strip().split()) if isinstance(x, str) else x))
    # return df.applymap(lambda x: x.strip().replace("  ", " ") if isinstance(x, str) else x)
    #return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # return df.applymap(lambda x: x.strip().replace("  ", " ") if isinstance(x, str) else x)
    # return df.apply(lambda x: x.str.strip().replace("  ", " ") if x.dtype == "object" else x)
    #return df.apply(lambda x: x.str.strip().replace("  ", " ") if x.dtype == "object" else x.apply(lambda y: str(y).strip().replace("  ", " ") if isinstance(y, str) else y))
    # return df.applymap(lambda x: " ".join(str(x).strip().split()) if isinstance(x, str) else x)


def scan_email_format(df):
    # Detect the column that contains "email" (case-insensitive)
    # print("Scanning email format...", df.columns)
    email_col = next((col for col in df.columns if 'email' in col.lower()), None)
    
    if email_col is None:
        print("No email column found in dataset.")
        return []

    # Email validation regex pattern
    # pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\.[a-zA-Z]{2,}$'

    # Find invalid email indices
    flagged_indices = df[~df[email_col].astype(str).str.match(pattern, na=False)].index.tolist()
    
    return flagged_indices

def scan_data_types(df):
    """
    Checks if column values match the inferred data type of the column, 
    including handling for boolean values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        List of flagged row indices where data types do not match.
    """
    flagged_indices = []

    for col in df.columns:
        inferred_type = df[col].dropna().apply(type).mode()[0]  # Infer data type
        # print(f"Column: {col}, Inferred Type: {inferred_type}")
        # Force "Index" column to be an integer
        if col.lower() == "index":
            invalid_rows = df[~df[col].astype(str).str.match(r'^\d+$', na=False)].index.tolist()

        # Integer column validation
        elif np.issubdtype(inferred_type, np.integer):
            invalid_rows = df[~df[col].astype(str).str.match(r'^\d+$', na=False)].index.tolist()

        # Float column validation
        elif np.issubdtype(inferred_type, np.floating):
            invalid_rows = df[~df[col].astype(str).str.match(r'^\d+(\.\d+)?$', na=False)].index.tolist()

        # Boolean column validation
        elif set(df[col].dropna().astype(str).str.strip().unique()).issubset({"True", "False", "1", "0"}):
            invalid_rows = df[~df[col].astype(str).str.strip().isin({"True", "False", "1", "0"})].index.tolist()

        # String column validation: Flag numeric values in string columns
        elif inferred_type == str:
            invalid_rows = df[df[col].apply(lambda x: isinstance(x, (int, float)))].index.tolist()

        else:
            print(f"Skipping unsupported type: {inferred_type} for column {col}")
            continue  

        flagged_indices.extend(invalid_rows)

    return list(set(flagged_indices))  # Remove duplicates


def scan_categorical_values(df):
    flagged_indices = []
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    if 'gender' in df.columns:
        valid_genders = ['male', 'female', 'm', 'f', 'other']
        flagged_indices.extend(df[~df['gender'].str.lower().isin(valid_genders)].index.tolist())

    return list(set(flagged_indices))

def fix_categorical_values(df):
    # Find the actual "gender" column, regardless of case
    gender_col = next((col for col in df.columns if col.lower() == 'gender'), None)

    if gender_col:
        valid_genders = ['male', 'female', 'm', 'f', 'other']
        df[gender_col] = df[gender_col].apply(lambda x: x if pd.notna(x) and str(x).lower() in valid_genders else "NA")

    return df


def scan_negative_values(df):
    flagged_indices = []
    for col in df.select_dtypes(include=['number']):
        # print("Scanning negative values in column:", col)
        invalid_indices = df[df[col] < 0].index.tolist()
        if invalid_indices:
            flagged_indices.extend(invalid_indices)
    return list(set(flagged_indices))


def scan_outliers(df):
    flagged_indices = []
    for col in df.select_dtypes(include=[np.number]):  # Only check numeric columns
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile Range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        if outlier_indices:
            flagged_indices = outlier_indices

    return flagged_indices


def scan_special_chars_in_ids(df):
    flagged_indices = []
    for col in df.columns:
        if 'id' in col.lower() and 'email' not in col.lower():  # Check for ID-like columns, but exclude emails
            # print("Scanning special characters in column:", col)
            invalid_indices = df[~df[col].astype(str).str.match(r'^[A-Za-z0-9]+$', na=False)].index.tolist()
            if invalid_indices:
                flagged_indices.extend(invalid_indices)  # Append indices instead of overwriting
    return list(set(flagged_indices))  # Remove duplicates


# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            return redirect(url_for('upload_file'))

    files = [{'name': f, 'date': datetime.fromtimestamp(os.path.getmtime(os.path.join(UPLOAD_FOLDER, f))).strftime('%Y-%m-%d %H:%M:%S')}
             for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    return render_template('saved_data.html', files=files)


@app.route('/autoquality/<file_name>')
def autoquality(file_name):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    session['uploaded_file'] = file_path

    df = pd.read_csv(file_path)
    return render_template('autoquality.html', 
                           tables=[df.to_html(classes='table', table_id='dataTable')], 
                           rules=RULES, file_name=file_name)


@app.route('/scan_rule/<file_name>/<rule_id>')
def scan_rule(file_name, rule_id):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    df = pd.read_csv(file_path)

    rule_functions = {
        'empty_rows': scan_empty_rows,
        'duplicates': scan_duplicates,
        'missing_values': scan_missing_values,
        'whitespace': scan_whitespace,
        'email_format': scan_email_format,
        'data_types': scan_data_types,
        'categorical_values': scan_categorical_values,
        'negative_values': scan_negative_values,
        'outliers': scan_outliers,
        'special_char': scan_special_chars_in_ids
        # # 'boolean_values': scan_boolean_values,
        # # 'contact_format': scan_contact_format
        # # 'data_inconsistency': check_data_inconsistency,
        # # 'ranges': check_ranges,
        # 'cross_field_validation': cross_field_validation,
        # 'normalize_data': normalize_data,
        # 'encode_categorical_data': encode_categorical_data,
        # 'anonymize_data': anonymize_data
    }

    result = rule_functions[rule_id](df) if rule_id in rule_functions else []
    quality_score = max(0, (len(df) - len(result)) / len(df) * 100) if isinstance(result, list) else 100

    # Save modified DataFrame if necessary
    # if rule_id in ['normalize_data', 'encode_categorical_data', 'anonymize_data']:
    #     modified_file_name = f"modified_{rule_id}_{file_name}"
    #     modified_file_path = os.path.join(UPLOAD_FOLDER, modified_file_name)
    #     pd.DataFrame(result).to_csv(modified_file_path, index=False)
    #     return jsonify({'message': f'Data modified and saved as {modified_file_name}', 'quality_score': quality_score})

    return jsonify({'flagged_indices': result, 'quality_score': quality_score})


@app.route('/fix_rule/<file_name>')
def fix_rule(file_name):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if not os.path.exists(file_path):
        return "File not found", 404

    df = pd.read_csv(file_path)

    #Fix for empty rows
    df_cleaned = fix_empty_rows(df)

    # Drop duplicates
    df_cleaned = fix_duplicates(df_cleaned)

    # Fill missing values with "NA"
    df_cleaned = fix_missing_values(df_cleaned)

    # Strip whitespace from string columns
    df_cleaned = fix_whitespace(df_cleaned)

    #Fix for categorical values
    df_cleaned = fix_categorical_values(df_cleaned)

    # #Fix for boolean values
    # # df_cleaned = fix_boolean_values(df_cleaned)

    cleaned_file_name = f"cleaned_{file_name}"
    clean_file_path = os.path.join(UPLOAD_FOLDER, cleaned_file_name)
    df_cleaned.to_csv(clean_file_path, index=False)

    return render_template('clean_data.html', 
                           tables=[df_cleaned.to_html(classes='table', table_id='dataTable')],
                           total_rows=len(df),
                           cleaned_rows=len(df_cleaned),
                           file_name=cleaned_file_name)


@app.route('/download/<file_name>')
def download(file_name):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)