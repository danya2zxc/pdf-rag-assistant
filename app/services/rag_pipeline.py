from typing import List
from openai import OpenAI
from ..core.config import settings







class RAGPipeline:
    def __init__(self, store, model="gpt-4o-mini", k=3):
        self.store = store
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.k = k

    def build_prompt(self, question: str, chunks: List[str]):
        context = "\n\n".join(chunks)
        
        return f"""Answer the question using only context below.
        If the context is irrelevant or insufficient, answer "I dont know."
        Context:
        {context}
        Question: {question}
        Answer:
        """

    def answer(self, question: str):
        results = self.store.search(question, k=self.k)
        if not results or results[0][1] > 1.5: 
            return {"question": question, "answer": "I don't know", "sources": []}

        chunks = [text for text, _ in results]
        prompt = self.build_prompt(question, chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": [
                {"text": text, "distance": dist}
                for text, dist in results
            ],
        }
