import streamlit as st
import streamlit.components.v1 as components
import os

# Set page configuration
st.set_page_config(
    page_title="AI Learning Hub | Class XII",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    # Hide Streamlit's default header, footer, and padding to maximize space
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 0rem;
                padding-right: 0rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Get the directory where app.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the HTML file
    html_file_path = os.path.join(BASE_DIR, "index.html")
    
    if os.path.exists(html_file_path):
        import re
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        def get_file_content(rel_path):
            abs_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(abs_path):
                with open(abs_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""

        # Inline CSS
        css_content = get_file_content("css/style.css")
        html_content = re.sub(r'<link\s+rel="stylesheet"\s+href="css/style\.css">', f'<style>{css_content}</style>', html_content)
        
        # Inline JS files
        curr_js = get_file_content("js/data/curriculum.js")
        html_content = re.sub(r'<script\s+src="js/data/curriculum\.js"></script>', f'<script>{curr_js}</script>', html_content)
        
        vis_js = get_file_content("js/components/visualizer.js")
        html_content = re.sub(r'<script\s+src="js/components/visualizer\.js"></script>', f'<script>{vis_js}</script>', html_content)
        
        main_js = get_file_content("js/main.js")
        html_content = re.sub(r'<script\s+src="js/main\.js"></script>', f'<script>{main_js}</script>', html_content)

        st.components.v1.html(html_content, height=1200, scrolling=True)
    else:
        st.error(f"Could not find the file: {html_file_path}. Please make sure you are running the app from the correct directory.")

if __name__ == "__main__":
    main()
