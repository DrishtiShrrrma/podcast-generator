import streamlit as st
import os
from openai import OpenAI
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader
)
from datetime import datetime
from pydub import AudioSegment
import pytz

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
import os
import tempfile
from datetime import datetime
import pytz


class DocumentRAG:
    def __init__(self):
        self.document_store = None
        self.qa_chain = None
        self.document_summary = ""
        self.chat_history = []
        self.last_processed_time = None
        self.api_key = os.getenv("OPENAI_API_KEY")  # Fetch the API key from environment variable
        self.init_time = datetime.now(pytz.UTC)

        if not self.api_key:
            raise ValueError("API Key not found. Make sure to set the 'OPENAI_API_KEY' environment variable.")

        # Persistent directory for Chroma to avoid tenant-related errors
        self.chroma_persist_dir = "./chroma_storage"
        os.makedirs(self.chroma_persist_dir, exist_ok=True)

    def process_documents(self, uploaded_files):
        """Process uploaded files by saving them temporarily and extracting content."""
        if not self.api_key:
            return "Please set the OpenAI API key in the environment variables."
        if not uploaded_files:
            return "Please upload documents first."

        try:
            documents = []
            for uploaded_file in uploaded_files:
                # Save uploaded file to a temporary location
                temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]).name
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                # Determine the loader based on the file type
                if temp_file_path.endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                elif temp_file_path.endswith('.txt'):
                    loader = TextLoader(temp_file_path)
                elif temp_file_path.endswith('.csv'):
                    loader = CSVLoader(temp_file_path)
                else:
                    return f"Unsupported file type: {uploaded_file.name}"

                # Load the documents
                try:
                    documents.extend(loader.load())
                except Exception as e:
                    return f"Error loading {uploaded_file.name}: {str(e)}"

            if not documents:
                return "No valid documents were processed. Please check your files."

            # Split text for better processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            documents = text_splitter.split_documents(documents)

            # Combine text for later summary generation
            self.document_text = " ".join([doc.page_content for doc in documents])  # Store for later use


            # Create embeddings and initialize retrieval chain
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            self.document_store = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=self.chroma_persist_dir  # Persistent directory for Chroma
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model_name='gpt-4', api_key=self.api_key),
                self.document_store.as_retriever(search_kwargs={'k': 6}),
                return_source_documents=True,
                verbose=False
            )

            self.last_processed_time = datetime.now(pytz.UTC)
            return "Documents processed successfully!"
        except Exception as e:
            return f"Error processing documents: {str(e)}"

    def generate_summary(self, text, language):
        """Generate a summary of the provided text in the specified language."""
        if not self.api_key:
            return "API Key not set. Please set it in the environment variables."
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"Summarize the document content concisely in {language}. Provide 3-5 key points for discussion."},
                    {"role": "user", "content": text[:4000]}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def create_podcast(self, language):
        """Generate a podcast script and audio based on doc summary in the specified language."""
        if not self.document_summary:
            return "Please process documents before generating a podcast.", None

        if not self.api_key:
            return "Please set the OpenAI API key in the environment variables.", None

        try:
            client = OpenAI(api_key=self.api_key)

            # Generate podcast script
            script_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a professional podcast producer. Create a natural dialogue in {language} based on the provided document summary."},
                    {"role": "user", "content": f"""Based on the following document summary, create a 1-2 minute podcast script:
                    1. Clearly label the dialogue as 'Host 1:' and 'Host 2:'
                    2. Keep the content engaging and insightful.
                    3. Use conversational language suitable for a podcast.
                    4. Ensure the script has a clear opening and closing.
                    Document Summary: {self.document_summary}"""}
                ],
                temperature=0.7
            )

            script = script_response.choices[0].message.content
            if not script:
                return "Error: Failed to generate podcast script.", None

            # Convert script to audio
            final_audio = AudioSegment.empty()
            is_first_speaker = True

            lines = [line.strip() for line in script.split("\n") if line.strip()]
            for line in lines:
                if ":" not in line:
                    continue

                speaker, text = line.split(":", 1)
                if not text.strip():
                    continue

                try:
                    voice = "nova" if is_first_speaker else "onyx"
                    audio_response = client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=text.strip()
                    )

                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    audio_response.stream_to_file(temp_audio_file.name)

                    segment = AudioSegment.from_file(temp_audio_file.name)
                    final_audio += segment
                    final_audio += AudioSegment.silent(duration=300)

                    is_first_speaker = not is_first_speaker
                except Exception as e:
                    print(f"Error generating audio for line: {text}")
                    print(f"Details: {e}")
                    continue

            if len(final_audio) == 0:
                return "Error: No audio could be generated.", None

            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            final_audio.export(output_file, format="mp3")
            return script, output_file

        except Exception as e:
            return f"Error generating podcast: {str(e)}", None

    def handle_query(self, question, history, language):
        """Handle user queries in the specified language."""
        if not self.qa_chain:
            return history + [("System", "Please process the documents first.")]
        try:
            preface = """
            Instruction: Respond in {language}. Be professional and concise, keeping the response under 300 words.
            If you cannot provide an answer, say: "I am not sure about this question. Please try asking something else."
            """
            query = f"{preface}\nQuery: {question}"

            result = self.qa_chain({
                "question": query,
                "chat_history": [(q, a) for q, a in history]
            })

            if "answer" not in result:
                return history + [("System", "Sorry, an error occurred.")]

            history.append((question, result["answer"]))
            return history
        except Exception as e:
            return history + [("System", f"Error: {str(e)}")]

# Initialize RAG system in session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = DocumentRAG()

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown(
        """
        This app is inspired by the [RAG_HW HuggingFace Space](https://huggingface.co/spaces/wint543/RAG_HW).  
        It allows users to upload documents, generate summaries, ask questions, and create podcasts.
        """
    )
    st.markdown("### Steps:")
    st.markdown("1. Upload documents.")
    st.markdown("2. Generate summaries.")
    st.markdown("3. Ask questions.")
    st.markdown("4. Create podcasts.")

# Streamlit UI
# Sidebar
#with st.sidebar:
    #st.title("About")
    #st.markdown(
        #"""
        #This app is inspired by the [RAG_HW HuggingFace Space](https://huggingface.co/spaces/wint543/RAG_HW).  
        #It allows users to:
        #1. Upload and process documents
        #2. Generate summaries
        #3. Ask questions
        #4. Create podcasts
        #"""
    #)

# Main App
st.title("Document Analyzer & Podcast Generator")

# Step 1: Upload and Process Documents
st.subheader("Step 1: Upload and Process Documents")
uploaded_files = st.file_uploader("Upload files (PDF, TXT, CSV)", accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        # Process the uploaded files
        result = st.session_state.rag_system.process_documents(uploaded_files)
        if "successfully" in result:
            st.success(result)
        else:
            st.error(result)
    else:
        st.warning("No files uploaded.")

    
# Step 2: Generate Summaries
st.subheader("Step 2: Generate Summaries")
st.write("Select Summary Language:")
summary_language_options = ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese"]
summary_language = st.radio(
    "", 
    summary_language_options, 
    horizontal=True, 
    key="summary_language"
)

if st.button("Generate Summary"):
    if hasattr(st.session_state.rag_system, "document_text") and st.session_state.rag_system.document_text:
        summary = st.session_state.rag_system.generate_summary(st.session_state.rag_system.document_text, summary_language)
        st.session_state.rag_system.document_summary = summary
        st.text_area("Document Summary", summary, height=200)
    else:
        st.info("Please process documents first to generate summaries.")


# Step 3: Ask Questions
st.subheader("Step 3: Ask Questions")
st.write("Select Q&A Language:")
qa_language_options = ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese"]
qa_language = st.radio(
    "", 
    qa_language_options, 
    horizontal=True, 
    key="qa_language"
)

if st.session_state.rag_system.qa_chain:
    history = []
    user_question = st.text_input("Ask a question:")
    if st.button("Submit Question"):
        # Handle the user query
        history = st.session_state.rag_system.handle_query(user_question, history, qa_language)
        for question, answer in history:
            st.chat_message("user").write(question)
            st.chat_message("assistant").write(answer)
else:
    st.info("Please process documents first to enable Q&A.")

# Step 4: Generate Podcast
st.subheader("Step 4: Generate Podcast")
st.write("Select Podcast Language:")
podcast_language_options = ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese"]
podcast_language = st.radio(
    "", 
    podcast_language_options, 
    horizontal=True, 
    key="podcast_language"
)

if st.session_state.rag_system.document_summary:
    if st.button("Generate Podcast"):
        script, audio_path = st.session_state.rag_system.create_podcast(podcast_language)
        if audio_path:
            st.text_area("Generated Podcast Script", script, height=200)
            st.audio(audio_path, format="audio/mp3")
            st.success("Podcast generated successfully! You can listen to it above.")
        else:
            st.error(script)
else:
    st.info("Please process documents and generate summaries before creating a podcast.")
