
import streamlit as st
import rag_on_files

# Access the functions defined in the notebook
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "xlsx", "txt"])

if uploaded_file is not None:
    st.write("File uploaded:", uploaded_file.name)

    with st.spinner("Processing..."):
        file_name = uploaded_file.name

        if file_name.endswith('.pdf'):
            text_data = rag_on_files.read_pdf(uploaded_file)
        elif file_name.endswith('.docx'):
            text_data = rag_on_files.read_docx(uploaded_file)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            text_data = df.to_string()
        elif file_name.endswith('.txt'):
            text_data = uploaded_file.read().decode('utf-8')
        else:
            st.error("Unsupported file type!")

        partitions = rag_on_files.partition_text(text_data, max_length=512)
        faiss_index, doc_ids = rag_on_files.store_in_faiss(partitions)

    query = st.text_input("Enter your query")
    if st.button("Ask"):
        if query:
            response = rag_on_files.ask(query)
            st.write(response)
        else:
            st.warning("Please enter a query")
