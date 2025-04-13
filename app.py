#!/usr/bin/env python3
import os
import uuid
import json
import matplotlib
import re

# Try to import markdown, fallback to basic HTML conversion if not available
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# Use non-interactive backend to avoid GUI issues
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from src.main import setup_client, detect_data_manipulation

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

def basic_markdown_to_html(md_text):
    """Simple markdown to HTML converter for fallback when markdown package isn't available"""
    html = md_text
    
    # Convert headers
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Convert code blocks
    html = re.sub(r'```json\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # Convert images
    html = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1">', html)
    
    # Convert bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert italics
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Convert paragraphs (simplified)
    html = '<p>' + html.replace('\n\n', '</p><p>') + '</p>'
    
    return f'<div class="markdown-content">{html}</div>'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'csv', 'dta'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Validate file type
    if not allowed_file(file.filename):
        flash(f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}')
        return redirect(request.url)
    
    try:
        # Create a unique folder for this analysis
        analysis_id = str(uuid.uuid4())
        analysis_folder = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Initialize Claude client
        client = setup_client()
        
        # Run analysis
        report = detect_data_manipulation(client, file_path, analysis_folder)
        
        # Store analysis information
        report_filename = f"report_{filename}.md"
        analysis_info = {
            'id': analysis_id,
            'filename': filename,
            'timestamp': os.path.getmtime(file_path),
            'report_path': os.path.join(analysis_folder, report_filename),
            'report_filename': report_filename,
        }
        
        # Save analysis metadata
        with open(os.path.join(analysis_folder, 'analysis_info.json'), 'w') as f:
            json.dump(analysis_info, f)
        
        return redirect(url_for('view_results', analysis_id=analysis_id))
    
    except Exception as e:
        flash(f'Error during analysis: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results/<analysis_id>')
def view_results(analysis_id):
    # Validate analysis ID
    analysis_folder = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
    if not os.path.exists(analysis_folder):
        flash('Analysis not found')
        return redirect(url_for('index'))
    
    # Load analysis info
    try:
        with open(os.path.join(analysis_folder, 'analysis_info.json'), 'r') as f:
            analysis_info = json.load(f)
        
        # Read report content
        report_file_path = os.path.join(analysis_folder, analysis_info.get('report_filename', f"report_{analysis_info['filename']}.md"))
        if not os.path.exists(report_file_path):
            report_file_path = analysis_info['report_path']
            
        with open(report_file_path, 'r') as f:
            report_content = f.read()
        
        # Extract manipulation rating if present
        manipulation_rating = None
        rating_match = re.search(r'MANIPULATION_RATING:\s*(\d+)', report_content)
        if rating_match:
            manipulation_rating = int(rating_match.group(1))
            # Remove the rating line from the report content for cleaner display
            report_content = re.sub(r'MANIPULATION_RATING:\s*\d+\s*\n', '', report_content)
        
        # Convert markdown to HTML
        if HAS_MARKDOWN:
            try:
                report_html = markdown.markdown(
                    report_content,
                    extensions=['fenced_code', 'tables']
                )
            except Exception as e:
                print(f"Error rendering markdown: {e}")
                report_html = basic_markdown_to_html(report_content)
        else:
            report_html = basic_markdown_to_html(report_content)
        
        # Fix image paths in HTML
        base_url = url_for("get_file", analysis_id=analysis_id, filename='')
        report_html = report_html.replace('src="', f'src="{base_url}')
        
        # Get image files
        image_files = [f for f in os.listdir(analysis_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        
        return render_template(
            'results.html',
            analysis=analysis_info,
            report=report_content,
            report_html=report_html,
            images=image_files,
            analysis_id=analysis_id,
            manipulation_rating=manipulation_rating
        )
    
    except Exception as e:
        flash(f'Error loading results: {str(e)}')
        return redirect(url_for('index'))

@app.route('/file/<analysis_id>/<path:filename>')
def get_file(analysis_id, filename):
    """Serve files from the results directory"""
    return send_from_directory(os.path.join(app.config['RESULTS_FOLDER'], analysis_id), filename)

if __name__ == '__main__':
    app.run(debug=True)