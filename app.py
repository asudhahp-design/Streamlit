import streamlit as st
import os
from dotenv import load_dotenv
from rag_utils import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

load_dotenv()

def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

    st.title("📄 AI-powered Document Assistant Chatbot")
    st.markdown("Upload your PDFs and ask questions about them!")

    # Initialize session state for chat history and vector store
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for PDF upload and API Key configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        
        st.header("Document Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process Documents"):
            if not api_key:
                st.error("Please provide an OpenAI API Key.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # 1. Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("Could not extract text from the provided PDFs.")
                    else:
                        # 2. Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Create Vector Store
                        vector_store = get_vector_store(text_chunks, api_key)
                        st.session_state.vector_store = vector_store
                        
                        st.success("Processing complete! You can now ask questions.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("Please provide your OpenAI API Key in the sidebar.")
            return
            
        if st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF first.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_chain = get_conversational_chain(st.session_state.vector_store, api_key)
                
                # Execute RAG chain
                response = rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                
                st.markdown(answer)
                
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
