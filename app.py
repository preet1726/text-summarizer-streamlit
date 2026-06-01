from functools import lru_cache
from typing import Optional

import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def fetch_text(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n\n".join(paragraphs)
    except Exception as e:
        st.error(f"Error fetching the URL: {e}")
        return None


def clean_and_truncate(text: str, max_chars: int = 20000) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "..."
    return cleaned


@lru_cache(maxsize=None)
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return tokenizer, model, device


class Summarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.tokenizer, self.model, self.device = load_model(model_name)
        self.model_name = model_name

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
    ) -> str:
        try:
            if "t5" in self.model_name.lower():
                inputs = "summarize: " + text
            else:
                inputs = text

            encoded = self.tokenizer(
                inputs,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            model_max_length = encoded["input_ids"].shape[1] + max_length

            output_tokens = self.model.generate(
                **encoded,
                max_length=model_max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=True,
            )

            summary = self.tokenizer.decode(
                output_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return summary.strip()
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return ""


def main():
    st.set_page_config(page_title="Text Summarizer App", layout="wide")

    st.title("Text Summarizer App")
    st.markdown("Summarize text, files and URLs")

    left, right = st.columns([1, 2])

    with left:
        input_mode = st.radio("Select input mode:", ("Text", "File", "URL"))

        model_choice = st.selectbox(
            "Model",
            [
                "sshleifer/distilbart-cnn-12-6",
                "facebook/bart-large-cnn",
                "t5-small",
            ],
        )

        min_length = st.number_input(
            "Minimum summary tokens",
            min_value=5,
            max_value=100,
            value=40,
        )

        max_length = st.number_input(
            "Maximum summary tokens",
            min_value=10,
            max_value=500,
            value=150,
        )

        do_sample = st.checkbox(
            "Use sampling (creative summaries)",
            value=False,
        )

        raw_text: Optional[str] = None
        uploaded_file = None
        url_input: Optional[str] = None

        if input_mode == "Text":
            raw_text = st.text_area(
                "Enter text to summarize",
                height=250,
            )

        elif input_mode == "File":
            uploaded_file = st.file_uploader(
                "Upload .txt or .md file",
                type=["txt", "md"],
            )

        elif input_mode == "URL":
            url_input = st.text_input("Enter URL")

        summarize_btn = st.button("Summarize")

    with right:
        st.subheader("Original Text")
        source_container = st.empty()

        st.subheader("Summary")
        summary_container = st.empty()

        if summarize_btn:
            user_input_text = ""

            if input_mode == "Text" and raw_text:
                user_input_text = raw_text

            elif input_mode == "File" and uploaded_file:
                uploaded_bytes = uploaded_file.read()

                try:
                    user_input_text = uploaded_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    user_input_text = uploaded_bytes.decode("latin-1")

            elif input_mode == "URL" and url_input:
                fetched = fetch_text(url_input)

                if not fetched:
                    st.warning("Could not retrieve text from the URL.")
                    st.stop()

                user_input_text = fetched

            if not user_input_text:
                st.warning("Please provide valid input text, file, or URL.")
                st.stop()

            user_input_text = clean_and_truncate(user_input_text)

            source_container.code(user_input_text[:8000])

            if min_length >= max_length:
                st.warning(
                    "Minimum token count must be less than maximum token count."
                )
                st.stop()

            summarizer = Summarizer(model_choice)

            with st.spinner("Generating summary..."):
                summary_text = summarizer.summarize(
                    user_input_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                )

            if not summary_text:
                st.warning("Summarization failed.")
                st.stop()

            summary_container.write(summary_text)


if __name__ == "__main__":
    main()
