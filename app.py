import streamlit as st
import pytesseract
from PIL import Image
import joblib
from transformers import pipeline
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime
import os
from pdf2image import convert_from_bytes

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(page_title="DocuMagic AI", page_icon="✨", layout="wide")

# Custom CSS for a Colorful Interface
st.markdown("""
<style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    
    /* Result Cards */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    
    .category-tag {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Custom Button Styling */
    div.stButton > button {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white;
        font-size: 20px;
        padding: 10px 30px;
        border-radius: 30px;
        border: none;
        width: 100%;
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }
    
    /* Titles */
    h1 {
        background: -webkit-linear-gradient(45deg, #0984e3, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATABASE MANAGEMENT (Auth & History) ---
DB_FILE = 'user_data.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Create History Table
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  timestamp TEXT, 
                  filename TEXT, 
                  category TEXT, 
                  summary TEXT)''')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

def add_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', 
                  (username, make_hashes(password)))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =? AND password = ?', 
              (username, make_hashes(password)))
    data = c.fetchall()
    conn.close()
    return data

def save_history(username, filename, category, summary):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO history (username, timestamp, filename, category, summary) VALUES (?,?,?,?,?)',
              (username, timestamp, filename, category, summary))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"SELECT timestamp, filename, category, summary FROM history WHERE username = '{username}' ORDER BY id DESC", conn)
    conn.close()
    return df

# Initialize DB on start
init_db()

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('xgb_model_new.pkl')
        vectorizer = joblib.load('tfidf_vectorizer_new.pkl')
        encoder = joblib.load('label_encoder_new.pkl')
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        return model, vectorizer, encoder, summarizer
    except Exception as e:
        return None, None, None, None

xgb_model, tfidf_vectorizer, label_encoder, summarizer = load_resources()

# --- 4. CORE LOGIC ---
def process_document(image=None, provided_text=None):
    # A. OCR logic
    if provided_text:
        text = provided_text
    elif image:
        text = pytesseract.image_to_string(image)
    else:
        return None, None, "No input provided"

    if not text.strip():
        return None, None, "No text found"
    
    # B. Classification
    text_vec = tfidf_vectorizer.transform([text])
    pred_idx = xgb_model.predict(text_vec)[0]
    category = label_encoder.inverse_transform([pred_idx])[0]
    
    # C. Summarization
    input_text = text[:3000] 
    if len(input_text) < 50:
        summary = "Text too short to summarize."
    else:
        # Adjusted max_length slightly for stability
        res = summarizer(input_text, max_length=130, min_length=30, do_sample=False, truncation=True)
        summary = res[0]['summary_text']
        
    return category, summary, text

# --- 5. MAIN UI ---

# Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

def main():
    if not st.session_state['logged_in']:
        # --- LOGIN / SIGNUP SCREEN ---
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.title("🔐 DocuMagic Login")
            choice = st.selectbox("Select Action", ["Login", "Sign Up"])
            
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            
            if choice == "Login":
                if st.button("Login"):
                    result = login_user(username, password)
                    if result:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("Incorrect Username or Password")
            else:
                if st.button("Create Account"):
                    if add_user(username, password):
                        st.success("Account Created! Please Login.")
                    else:
                        st.warning("Username already taken.")

    else:
        # --- LOGGED IN DASHBOARD ---
        
        # Sidebar
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
            st.write(f"Welcome, **{st.session_state['username']}**!")
            if st.button("Log Out"):
                st.session_state['logged_in'] = False
                st.rerun()
            
            st.divider()
            st.subheader("📜 Your History")
            history_df = get_user_history(st.session_state['username'])
            if not history_df.empty:
                st.dataframe(history_df[['timestamp', 'category', 'filename']], hide_index=True)
            else:
                st.write("No history yet.")

        # Main Content
        st.title("✨ DocuMind AI")
        st.write("Upload a document image or PDF to automatically classify and summarize it.")
        
        # Updated uploader to accept PDF
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'tiff', 'pdf'], label_visibility="collapsed")
        
        if uploaded_file:
            col_img, col_res = st.columns([1, 1.5])
            
            # Variables for processing
            display_image = None
            extracted_text = None

            # --- DETECT FILE TYPE ---
            if uploaded_file.type == "application/pdf":
                with st.spinner("📄 Converting PDF pages..."):
                    try:
                        # Convert PDF to images
                        images = convert_from_bytes(uploaded_file.read())
                        
                        # Use first page for display
                        display_image = images[0]
                        
                        # OCR all pages for text analysis
                        full_text_list = []
                        for page_img in images:
                            full_text_list.append(pytesseract.image_to_string(page_img))
                        extracted_text = " ".join(full_text_list)
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
            else:
                # Standard Image Processing
                display_image = Image.open(uploaded_file)
                extracted_text = None # process_document will handle the OCR

            # --- DISPLAY & ANALYZE ---
            if display_image:
                with col_img:
                    st.image(display_image, caption="Uploaded Document Preview", use_column_width=True)
                
                with col_res:
                    st.write("## Ready to Analyze")
                    
                    if st.button("🚀 Process Document Now"):
                        if xgb_model is None:
                            st.error("Models failed to load.")
                        else:
                            with st.spinner("✨ Reading... Thinking... Writing..."):
                                # Call process_document with optional text (if PDF)
                                category, summary, raw_text = process_document(image=display_image, provided_text=extracted_text)
                                
                                if category:
                                    # Save to DB
                                    save_history(st.session_state['username'], uploaded_file.name, category, summary)
                                    
                                    # DISPLAY RESULTS (Pretty Card)
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <h3>📂 Document Type</h3>
                                        <span class="category-tag">{category.upper()}</span>
                                        <br><br>
                                        <h3>📝 Document Summary</h3>
                                        <p style="font-size: 1.1rem; color: #555; line-height: 1.6;">{summary}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.success("Analysis Complete & Saved to History!")
                                    
                                    with st.expander("View Full Extracted Text"):
                                        st.write(raw_text)
                                    
                                else:
                                    st.error("Could not read text from document.")

if __name__ == '__main__':
    main()
