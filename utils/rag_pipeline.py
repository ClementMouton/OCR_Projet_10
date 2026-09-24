import logging
from typing import Any

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from .config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from .vector_store import VectorStoreManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)


SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.

Ta mission est de répondre aux questions des fans en animant le débat.

---

{context_str}

---

QUESTION DU FAN:

{question}

RÉPONSE DE L'ANALYSTE NBA:"""


class RAGPipeline:
    def __init__(self):
        if not MISTRAL_API_KEY:
            raise ValueError(
                "La clé API Mistral (MISTRAL_API_KEY) n'est pas définie."
            )

        self.client = MistralClient(api_key=MISTRAL_API_KEY)
        self.model = MODEL_NAME
        self.vector_store = VectorStoreManager()

        if self.vector_store.index is None or not self.vector_store.document_chunks:
            raise RuntimeError(
                "L'index FAISS ou les chunks n'ont pas pu être chargés."
            )

        logging.info(
            "RAGPipeline initialisé avec %s vecteurs.",
            self.vector_store.index.ntotal
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        return self.vector_store.search(
            question,
            k=SEARCH_K
        )

    def generate(self, question: str, search_results: list[dict[str, Any]]) -> str:
        context_str = "\n\n---\n\n".join(
            [
                (
                    f"Source: {result['metadata'].get('source', 'Inconnue')} "
                    f"(Score: {result['score']:.1f}%)\n"
                    f"Contenu: {result['text']}"
                )
                for result in search_results
            ]
        )

        if not search_results:
            context_str = (
                "Aucune information pertinente trouvée dans la base "
                "de connaissances pour cette question."
            )

        final_prompt = SYSTEM_PROMPT.format(
            context_str=context_str,
            question=question
        )

        messages = [
            ChatMessage(
                role="user",
                content=final_prompt
            )
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0.1
        )

        if not response.choices:
            raise RuntimeError(
                "L'API Mistral n'a retourné aucune réponse valide."
            )

        return response.choices[0].message.content

    def ask(self, question: str) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        search_results = self.retrieve(question)

        answer = self.generate(
            question=question,
            search_results=search_results
        )

        return {
            "question": question,
            "answer": answer,
            "contexts": [
                result["text"]
                for result in search_results
            ],
            "sources": [
                result["metadata"].get("source", "Inconnue")
                for result in search_results
            ],
            "scores": [
                result["score"]
                for result in search_results
            ]
        }