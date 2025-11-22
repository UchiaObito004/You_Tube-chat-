import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------
# LOAD API KEY
# -------------------------------------------------------
api_key = st.secrets["GOOGLE_API_KEY"]
ytt_api=YouTubeTranscriptApi()
# -------------------------------------------------------
# INITIALIZE CHAT HISTORY
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🎥 YouTube Video Q&A App with Chat History")

video_id = st.text_input("Enter YouTube Video ID")

# -------------------------------------------------------
# PROCESS VIDEO
# -------------------------------------------------------
if st.button("Process Video"):
    try:
        st.info("Fetching transcript...")

        transcript_list = ytt_api.fetch(
            video_id,
            languages=["en", "hi"]
        )

        st.success("Transcript fetched successfully!")

        transcript = " ".join(snippet.text for snippet in transcript_list)

        st.info("Splitting transcript...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        )
        docs = splitter.create_documents([transcript])

        st.info("Generating embeddings...")
        embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )

        db = FAISS.from_documents(docs, embedder)

        st.session_state["db"] = db
        st.session_state["transcript"] = transcript
        st.session_state["chat_history"] = []   # Reset chat

        st.success("Video processed! You can now ask questions.")

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript available in English or Hindi.")
    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------------------------------
# FALLBACK PROMPT (General Answer)
# -------------------------------------------------------
fallback_prompt = """
You are a helpful, knowledgeable assistant.

If the user's question is NOT related to the video content:
→ Give a complete, helpful general answer.
→ DO NOT say things like:
   "I cannot see the video" or "provide a link" etc.

Just answer normally.

Question:
{q}

Answer:
"""

# -------------------------------------------------------
# QUESTION INPUT
# -------------------------------------------------------
st.subheader("Ask a Question About the Video")
question = st.text_input("Your question")

# -------------------------------------------------------
# ANSWER BUTTON
# -------------------------------------------------------
if st.button("Get Answer"):

    if "db" not in st.session_state:
        st.error("Please process a video first.")
        st.stop()

    try:
        db = st.session_state["db"]
        docs = db.similarity_search(question, k=4)
        context_text = " ".join(d.page_content for d in docs).strip()

        # ---------------------------------------------------
        # If no relevant context → fallback answer
        # ---------------------------------------------------
        if context_text == "":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.7
            )
            reply = llm.invoke(fallback_prompt.format(q=question)).content

            # Save history
            st.session_state["chat_history"].append(("You", question))
            st.session_state["chat_history"].append(("Assistant", reply))

            st.write(reply)
            st.stop()

        # ---------------------------------------------------
        # Prompt for video-related answer
        # ---------------------------------------------------
        template = PromptTemplate.from_template("""
Use ONLY the video context below to answer.
If answer is NOT in the context, respond EXACTLY: "NOT_FOUND"

Context:
{context}

Question:
{question}

Answer:
""")

        chain = template | ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=api_key,
            temperature=0.2
        )

        response = chain.invoke({
            "context": context_text,
            "question": question
        })

        answer = response.content

        # If model signals NOT_FOUND → fallback
        if "NOT_FOUND" in answer:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.7
            )
            answer = llm.invoke(fallback_prompt.format(q=question)).content

        # Save history
        st.session_state["chat_history"].append(("You", question))
        st.session_state["chat_history"].append(("Assistant", answer))

        st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------------------------------
st.subheader("📝 Chat History")

for role, msg in st.session_state["chat_history"]:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg}")
