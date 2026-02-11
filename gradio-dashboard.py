import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


import gradio as gr

import os
#
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_openai_key():
    # Docker secrets
    secret_path = "/run/secrets/openai_key"
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            return f.read().strip()

    # Fallback to env / .env (local dev)
    return os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = load_openai_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

books = pd.read_csv("books_with_emotions.csv")


books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "no_cover.jpg",
    books["large_thumbnail"],
)


raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size= 1 , chunk_overlap=0, separator="\n")
document = text_splitter.split_documents(raw_documents)
# db_books = Chroma.from_documents(document,OpenAIEmbeddings())

db_books = Chroma.from_documents(
    document,
    OpenAIEmbeddings(),
    persist_directory="./chroma_data"
)



def retrieve_semantic_recommendations(
        query : str ,
        category: str = None ,
        tone: str = None,
        initial_top_k : int = 50,
        final_top_k : int = 16,
) -> pd.DataFrame:
    #
    # recs = db_books.similarity_search_with_score(query , k=initial_top_k)
    # books_list = [int(rec.page_content.strip("""""").split()[0]) for rec in recs]

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(doc.page_content.strip().split()[0].strip('"')) for doc, score in recs]

    # book_recs = books[books["isbn13"].isin(books_list).head(final_top_k)]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_category"] == category].head(final_top_k)
    else:
        book_recs =book_recs.head(final_top_k)

    if tone =="Happy":
        book_recs.sort_values(by="Joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query : str ,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _,row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"

        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# Your categories and tones
categories = ["All", "Fiction", "Nonfiction"]
tones = ["All", "Happy", "Surprise", "Angry", "Suspenseful", "Sad"]

# Custom CSS for rounded button and nicer look
css = """
.gr-button {
    border-radius: 9999px !important;  /* super rounded */
    font-size: 18px !important;
    padding: 16px 32px !important;
    font-weight: bold !important;
}
.gr-textbox, .gr-dropdown {
    border-radius: 12px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gr-button {
        border-radius: 9999px !important;
        font-size: 18px !important;
        padding: 16px 32px !important;
        font-weight: bold !important;
    }
    .gr-textbox, .gr-dropdown {
        border-radius: 12px !important;
    }
""") as dashboard:
    gr.Markdown(
        """
        # 📚✨ Semantic Book Recommender ✨📚
        *Discover books by meaning, genre, and emotion*
        """
    )

    with gr.Column():
        user_query = gr.Textbox(
            label="🔍 What kind of book are you looking for?",
            placeholder="e.g., A story about second chances, family, and hope in a small town",
            lines=3
        )

        with gr.Row(equal_height=True):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📖 Genre",
                value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="❤️ Mood",
                value="All"
            )

        submit_button = gr.Button(
            "🚀 Find My Perfect Books! 🚀",
            variant="primary",
            size="lg"
        )

    gr.Markdown("## 🌟 Your Personalized Recommendations 🌟")

    output = gr.Gallery(
        label="Recommended Books",
        columns=5,
        rows=4,
        height="auto",
        object_fit="cover"
    )

    # No loading message — Gradio shows a spinner automatically during processing
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # share=True gives you a public link
        show_error=True
    )


