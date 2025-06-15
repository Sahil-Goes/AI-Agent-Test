import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Load LLM
model = OllamaLLM(model="llama3.2")

# Prompt template
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit UI
st.set_page_config(page_title="Pizza QA Bot", page_icon="🍕")
st.title("🍕 Pizza Restaurant Q&A Bot")
st.write("Ask a question about the pizza restaurant based on customer reviews.")

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Searching reviews and generating answer..."):
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})

        st.subheader("📝 Answer")
        st.write(result)

        st.subheader("📚 Top 5 Relevant Reviews")
        for i, review in enumerate(reviews, 1):
            st.markdown(f"**Review {i}:** {review.page_content}")
