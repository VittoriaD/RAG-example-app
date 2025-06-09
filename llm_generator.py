import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    def generate(self, context, question, model="gpt-4"):
        prompt = f"""You are a helpful assistant. Use only the following context to answer the question. 
        Do not use any external knowledge or make assumptions.

        If the context does not contain enough information, say "I don't know based on the provided context."

        Context:
        {context}

        Question: {question}

        Answer (based only on the context above):"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
