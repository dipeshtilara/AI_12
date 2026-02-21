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
    st.title("🎓 AI Learning Hub - Class XII")
    st.write("Welcome to the Streamlit wrapper for the AI Learning Hub. You can view the web application below or run it directly in your browser.")
    
    # Path to the HTML file
    html_file_path = "index.html"
    
    if os.path.exists(html_file_path):
        # We use an iframe to render the static site. 
        # Note: Depending on Streamlit's environment, local CSS and JS files linked in the HTML 
        # might need to be in a static folder or hosted if they don't load correctly in the iframe.
        # Streamlit 1.28+ supports a `static` folder out of the box.
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.error(f"Could not find the file: {html_file_path}. Please make sure you are running the app from the correct directory.")

if __name__ == "__main__":
    main()
