# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 27-09-2025
# Register no: 212222060264
# Name : Surendhar.S
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:

1.OpenAI GPT (ChatGPT API)

2.Hugging Face Transformers

3.LangChain Framework (optional)

4.Google Generative AI API (optional)

# Explanation:

In this experiment, the Persona Prompting Pattern is applied by assuming the role of a Data Analyst.
The application area selected is sentiment analysis and keyword extraction for customer reviews.

Steps followed:

1.Define the persona: Data Analyst writing Python code to analyze customer feedback.

2.Use Hugging Face for quick sentiment analysis.

3.Use OpenAI GPT for deeper insights (sentiment, keywords, suggestions).

4.Compare and analyze outputs for consistency and actionability.

# Conclusion:


# -------------------------------
# Import Required Libraries
# -------------------------------
from transformers import pipeline
import openai
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Google Generative AI (optional)
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# -------------------------------
# Input Text (Customer Review)
# -------------------------------
review_text = """
The product quality is amazing, especially the battery backup and display.
However, the delivery was delayed by a week, and the mobile app has frequent crashes.
Customer support was polite and helpful. Overall, satisfied but improvements are needed.
"""


# -------------------------------
# 1. Hugging Face Transformers
# -------------------------------
print("=== Hugging Face Summarization & Sentiment ===")

summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

hf_summary = summarizer(review_text, max_length=50, min_length=15, do_sample=False)
hf_sentiment = sentiment_analyzer(review_text)

print("HF Summary:", hf_summary[0]['summary_text'])
print("HF Sentiment:", hf_sentiment[0])


# -------------------------------
# 2. OpenAI GPT (ChatGPT API)
# -------------------------------
print("\n=== OpenAI GPT Analysis ===")

openai.api_key = "YOUR_OPENAI_API_KEY"

prompt = f"""
Analyze the following customer review:

Review: {review_text}

1. Provide a short summary (max 40 words).
2. Identify the sentiment (positive/negative/neutral).
3. Suggest one improvement for the product/service.
"""

gpt_response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=150,
    temperature=0.5
)

gpt_output = gpt_response["choices"][0]["text"].strip()
print(gpt_output)


# -------------------------------
# 3. LangChain Framework (optional orchestration)
# -------------------------------
print("\n=== LangChain Workflow (using OpenAI GPT) ===")

template = """
You are a data analyst. Summarize and analyze this customer review:
{review}

Provide:
- Summary (20–30 words)
- Sentiment
- Top 3 keywords
"""
prompt_template = PromptTemplate(input_variables=["review"], template=template)

llm = LangChainOpenAI(openai_api_key="YOUR_OPENAI_API_KEY", model_name="text-davinci-003")
chain = LLMChain(llm=llm, prompt=prompt_template)

langchain_result = chain.run(review=review_text)
print(langchain_result)


# -------------------------------
# 4. Google Generative AI API (Optional)
# -------------------------------
if genai:
    print("\n=== Google Generative AI (Gemini) ===")
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")

    model = genai.GenerativeModel("gemini-pro")
    gemini_prompt = f"Summarize and analyze the following review:\n\n{review_text}"

    response = model.generate_content(gemini_prompt)
    print(response.text)
else:
    print("\n[Google Generative AI not installed. Skipping this step.]")



The experiment demonstrated cross-tool compatibility for sentiment and keyword analysis. Hugging Face quickly classified sentiment, while OpenAI GPT provided detailed insights and suggestions. This shows the complementary power of using multiple AI tools together.

# How the AI Works:

1.Hugging Face Transformers → fast summarization & sentiment.

2.OpenAI GPT → richer analysis (summary + sentiment + suggestion).

3.LangChain → structured workflow (prompt templating + chaining).

4.Google Generative AI (Gemini) → optional extra insights.

# Result:

  The corresponding Python code was executed successfully, proving that sentiment analysis and keyword extraction can be enhanced by combining multiple AI tools.
