# Text Summarization Tool - End-to-End Project

# Required libraries
import streamlit as st
from transformers import pipeline
import tensorflow as tf

# Define a mapping of languages to summarization models
language_model_map = {
    "English": "facebook/bart-large-cnn",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "Italian": "Helsinki-NLP/opus-mt-en-it",
    "Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "Russian": "Helsinki-NLP/opus-mt-en-ru",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "Japanese": "Helsinki-NLP/opus-mt-en-ja",
    "Korean": "Helsinki-NLP/opus-mt-en-ko",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Arabic": "Helsinki-NLP/opus-mt-en-ar",
    "Bengali": "Helsinki-NLP/opus-mt-en-bn",
    "Urdu": "Helsinki-NLP/opus-mt-en-ur",
    "Swahili": "Helsinki-NLP/opus-mt-en-sw",
    "Zulu": "Helsinki-NLP/opus-mt-en-zu",
    "Turkish": "Helsinki-NLP/opus-mt-en-tr",
    "Vietnamese": "Helsinki-NLP/opus-mt-en-vi",
    "Thai": "Helsinki-NLP/opus-mt-en-th",
    "Greek": "Helsinki-NLP/opus-mt-en-el",
    "Hebrew": "Helsinki-NLP/opus-mt-en-he",
    "Polish": "Helsinki-NLP/opus-mt-en-pl",
    "Czech": "Helsinki-NLP/opus-mt-en-cs",
    "Hungarian": "Helsinki-NLP/opus-mt-en-hu",
    "Romanian": "Helsinki-NLP/opus-mt-en-ro",
    "Bulgarian": "Helsinki-NLP/opus-mt-en-bg",
    "Indonesian": "Helsinki-NLP/opus-mt-en-id",
    "Malay": "Helsinki-NLP/opus-mt-en-ms",
    "Filipino": "Helsinki-NLP/opus-mt-en-tl",
    "Tamil": "Helsinki-NLP/opus-mt-en-ta",
    "Telugu": "Helsinki-NLP/opus-mt-en-te",
    "Kannada": "Helsinki-NLP/opus-mt-en-kn",
    "Marathi": "Helsinki-NLP/opus-mt-en-mr",
    "Gujarati": "Helsinki-NLP/opus-mt-en-gu",
    "Punjabi": "Helsinki-NLP/opus-mt-en-pa",
    "Sinhala": "Helsinki-NLP/opus-mt-en-si",
    "Nepali": "Helsinki-NLP/opus-mt-en-ne",
    "Pashto": "Helsinki-NLP/opus-mt-en-ps",
    "Farsi": "Helsinki-NLP/opus-mt-en-fa",
    "Serbian": "Helsinki-NLP/opus-mt-en-sr",
    "Croatian": "Helsinki-NLP/opus-mt-en-hr",
    "Slovak": "Helsinki-NLP/opus-mt-en-sk",
    "Slovenian": "Helsinki-NLP/opus-mt-en-sl",
    "Ukrainian": "Helsinki-NLP/opus-mt-en-uk",
    "Kazakh": "Helsinki-NLP/opus-mt-en-kk",
    "Uzbek": "Helsinki-NLP/opus-mt-en-uz",
    "Mongolian": "Helsinki-NLP/opus-mt-en-mn",
    "Amharic": "Helsinki-NLP/opus-mt-en-am",
    "Somali": "Helsinki-NLP/opus-mt-en-so",
    "Hausa": "Helsinki-NLP/opus-mt-en-ha",
    "Yoruba": "Helsinki-NLP/opus-mt-en-yo",
    "Igbo": "Helsinki-NLP/opus-mt-en-ig",
    "Afrikaans": "Helsinki-NLP/opus-mt-en-af",
    "Norwegian": "Helsinki-NLP/opus-mt-en-no",
    "Swedish": "Helsinki-NLP/opus-mt-en-sv",
    "Danish": "Helsinki-NLP/opus-mt-en-da",
    "Finnish": "Helsinki-NLP/opus-mt-en-fi",
    "Icelandic": "Helsinki-NLP/opus-mt-en-is",
    "Welsh": "Helsinki-NLP/opus-mt-en-cy",
    "Irish": "Helsinki-NLP/opus-mt-en-ga",
    "Scottish Gaelic": "Helsinki-NLP/opus-mt-en-gd",
    "Basque": "Helsinki-NLP/opus-mt-en-eu",
    "Catalan": "Helsinki-NLP/opus-mt-en-ca",
    "Galician": "Helsinki-NLP/opus-mt-en-gl",
    "Estonian": "Helsinki-NLP/opus-mt-en-et",
    "Latvian": "Helsinki-NLP/opus-mt-en-lv",
    "Lithuanian": "Helsinki-NLP/opus-mt-en-lt",
    "Macedonian": "Helsinki-NLP/opus-mt-en-mk",
    "Albanian": "Helsinki-NLP/opus-mt-en-sq",
    "Bosnian": "Helsinki-NLP/opus-mt-en-bs",
    "Armenian": "Helsinki-NLP/opus-mt-en-hy",
    "Georgian": "Helsinki-NLP/opus-mt-en-ka",
    "Azerbaijani": "Helsinki-NLP/opus-mt-en-az",
    "Kurdish": "Helsinki-NLP/opus-mt-en-ku",
    "Tajik": "Helsinki-NLP/opus-mt-en-tg",
    "Turkmen": "Helsinki-NLP/opus-mt-en-tk",
    "Kyrgyz": "Helsinki-NLP/opus-mt-en-ky",
    "Malayalam": "Helsinki-NLP/opus-mt-en-ml",
    "Lao": "Helsinki-NLP/opus-mt-en-lo",
    "Khmer": "Helsinki-NLP/opus-mt-en-km",
    "Burmese": "Helsinki-NLP/opus-mt-en-my",
    "Tibetan": "Helsinki-NLP/opus-mt-en-bo",
    "Maori": "Helsinki-NLP/opus-mt-en-mi",
    "Samoan": "Helsinki-NLP/opus-mt-en-sm",
    "Tongan": "Helsinki-NLP/opus-mt-en-to",
    "Hawaiian": "Helsinki-NLP/opus-mt-en-haw",
    "Esperanto": "Helsinki-NLP/opus-mt-en-eo",
    "Latin": "Helsinki-NLP/opus-mt-en-la",
    "Luxembourgish": "Helsinki-NLP/opus-mt-en-lb",
    "Corsican": "Helsinki-NLP/opus-mt-en-co",
    "Frisian": "Helsinki-NLP/opus-mt-en-fy",
    "Haitian Creole": "Helsinki-NLP/opus-mt-en-ht",
    "Javanese": "Helsinki-NLP/opus-mt-en-jv",
    "Sundanese": "Helsinki-NLP/opus-mt-en-su"
}

# Streamlit UI
st.set_page_config(page_title="Text Summarization Tool", layout="centered")
st.title("\U0001F4DD Real-Time Text Summarization Tool")

# Text input
user_input = st.text_area("Enter the text you want to summarize:", height=300)

# Sidebar options
st.sidebar.header("Customization Options")
language = st.sidebar.selectbox("Select Language:", list(language_model_map.keys()))
summary_size = st.sidebar.slider("Summary Size (Number of Words):", min_value=50, max_value=500, value=200, step=10)
summarization_style = st.sidebar.selectbox("Summarization Style:", ["Bullet Points", "Paragraph", "Custom"])

# Initialize summarization pipeline with selected language model
if language in language_model_map:
    summarizer = pipeline("summarization", model=language_model_map[language], framework="tf")
else:
    st.error(f"Summarization for {language} is not supported yet.")

# Summary button
if st.button("Summarize", key="main_summarize_button"):
    if user_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            # Adjust max_length and min_length based on summary size and style
            max_length = summary_size
            min_length = max(50, summary_size // 2)  # Ensure a reasonable minimum length

            # Generate summary
            summary = summarizer(user_input, max_length=max_length, min_length=min_length, do_sample=False)

            # Display summary based on style
            st.subheader("Summary:")
            if summarization_style == "Bullet Points":
                st.markdown("- " + summary[0]['summary_text'].replace(". ", ".\n- "))
            elif summarization_style == "Paragraph":
                st.success(summary[0]['summary_text'])
            elif summarization_style == "Custom":
                st.text_area("Custom Summary:", value=summary[0]['summary_text'], height=200)
                
                # Example output section
st.markdown("### Example Input:")
st.code("""
Artificial intelligence (AI) refers to the simulation of human intelligence in machines 
that are programmed to think like humans and mimic their actions. The term may also be 
applied to any machine that exhibits traits associated with a human mind such as learning 
and problem-solving.
""", language='text')

st.markdown("### Example Output:")
st.success("Artificial intelligence simulates human intelligence in machines, enabling them to learn and solve problems like humans.")
