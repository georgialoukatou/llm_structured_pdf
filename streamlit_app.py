import streamlit as st
from helpers import *  # Updated the import to "helpers" for variation
import base64

# Ensure API key is stored in the session state
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''


def show_pdf(uploaded_file):
    """
    Render a PDF file uploaded through Streamlit.

    The PDF is displayed within an iframe, set to 700x1000 pixels.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded PDF document.

    Returns
    -------
    None
    """
    # Convert file to bytes
    file_bytes = uploaded_file.getvalue()

    # Encode to Base64
    pdf_base64 = base64.b64encode(file_bytes).decode('utf-8')

    # Embed PDF as HTML
    pdf_viewer = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display in Streamlit
    st.markdown(pdf_viewer, unsafe_allow_html=True)


def create_streamlit_page():
    """
    Build the Streamlit interface with a two-column layout.

    - Left column: API key input and PDF uploader.
    - Right column: A brief introduction and explanation of the tool's purpose.

    Returns:
    -------
        col1: Left column object in Streamlit.
        col2: Right column object in Streamlit.
        uploaded_file: Uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="AI Document Tool")

    # Split layout into two equal columns
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Enter your OpenAI API Key")
        st.text_input('API Key (OpenAI)', type='password', key='api_key',
                      label_visibility="collapsed", disabled=False)
        st.header("Upload your document")
        uploaded_file = st.file_uploader("Upload a PDF file:", type="pdf")

    return col1, col2, uploaded_file


# Set up Streamlit interface
col1, col2, uploaded_file = create_streamlit_page()

# Handle uploaded file
if uploaded_file is not None:
    with col2:
        show_pdf(uploaded_file)

    # Extract text from the uploaded file
    file_content = extract_pdf_text(uploaded_file)
    st.session_state['vector_store'] = build_vector_store(file_content, 
                                                          api_key=st.session_state['api_key'], 
                                                          file_name=uploaded_file.name)
    st.write("File Uploaded and Processed Successfully")

# Generate results
with col1:
    if st.button("Create Summary Table"):
        with st.spinner("Processing your request..."):
            # Retrieve the vector store and query it
            response = query_vector_store(vectorstore=st.session_state['vector_store'],
                                          query="Provide the title, summary, publication date, and authors of the research document.",
                                          api_key=st.session_state['api_key'])
            
            placeholder = st.empty()
            placeholder.write(response)