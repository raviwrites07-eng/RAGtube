"""
app.py
=======
A simple local web page for asking questions about your transcribed playlist.

REQUIREMENTS (install on your laptop):
    pip install streamlit

RUN IT:
    streamlit run app.py

This opens a browser tab automatically (usually at http://localhost:8501).
Type a question, click the button, see the answer and clickable sources below it.
Leave the terminal window open while using the page -- closing it stops the app.
"""

import streamlit as st
from rag_core import answer_question

st.set_page_config(page_title="Playlist Q&A", page_icon="🎥")

st.title("🎥 Ask your playlist")
st.caption("Ask a question about the videos you've transcribed. Answers are grounded in the actual transcript content, with clickable timestamps.")

question = st.text_input("Your question", placeholder="e.g. what is a tokenizer?")
ask_clicked = st.button("Search", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching transcripts and generating answer..."):
        try:
            result = answer_question(question.strip())
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    st.caption(f"Searched across {result['total_chunks']} transcript chunks. Click a link to jump to that moment in the video.")

    for src in result["sources"]:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{src['video']}**  \n`{src['start']} - {src['end']}`  ·  match score {src['score']:.2f}")
                st.write(src["text"])
            with col2:
                if src["link"]:
                    st.link_button("▶ Watch", src["link"])
                else:
                    st.caption("No link available (re-run transcribe_playlist.py)")

elif ask_clicked:
    st.warning("Type a question first.")