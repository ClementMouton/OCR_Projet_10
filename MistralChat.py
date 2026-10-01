import logging

import streamlit as st

from utils.config import APP_TITLE, MODEL_NAME, NAME
from utils.rag_pipeline import RAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)


# ============================================================
# Configuration Streamlit
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏀",
)


# ============================================================
# Chargement du pipeline
# ============================================================

@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


try:
    pipeline = get_pipeline()

except Exception as error:
    st.error(
        "Impossible d'initialiser le pipeline NBA Analyst AI."
    )
    st.exception(error)
    st.stop()


# ============================================================
# Interface
# ============================================================

st.title(APP_TITLE)

st.caption(
    f"Assistant virtuel pour {NAME} | "
    f"Modèle : {MODEL_NAME}"
)


# ============================================================
# Historique
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis votre analyste IA pour la NBA. "
                "Je peux analyser les discussions du corpus, "
                "interroger les statistiques disponibles ou "
                "combiner les deux."
            ),
        }
    ]


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )


# ============================================================
# Question utilisateur
# ============================================================

question = st.chat_input(
    "Posez votre question sur la NBA..."
)


if question:

    # --------------------------------------------------------
    # Affichage question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyse de la question..."
        ):

            try:

                result = pipeline.ask(
                    question
                )

                answer = result[
                    "answer"
                ]

                st.write(answer)


                # --------------------------------------------
                # Informations techniques
                # --------------------------------------------

                with st.expander(
                    "Détails de l'analyse"
                ):

                    st.write(
                        f"**Route :** "
                        f"`{result['route']}`"
                    )


                    # Sources documentaires

                    if result["sources"]:

                        st.write(
                            "**Sources documentaires :**"
                        )

                        for source, score in zip(
                            result["sources"],
                            result["scores"],
                        ):

                            st.write(
                                f"- {source} "
                                f"({score:.1f} %)"
                            )


                    # SQL

                    if result["sql"]:

                        st.write(
                            "**Requête SQL :**"
                        )

                        st.code(
                            result["sql"],
                            language="sql",
                        )


                    # Résultats SQL

                    if result["sql_results"]:

                        st.write(
                            "**Résultats SQL :**"
                        )

                        st.json(
                            result["sql_results"]
                        )


                    # Erreur SQL

                    if result["sql_error"]:

                        st.warning(
                            "Le SQL Tool a rencontré "
                            "une erreur : "
                            f"{result['sql_error']}"
                        )


            except Exception as error:

                logging.exception(
                    "Erreur pendant le traitement "
                    "de la question."
                )

                answer = (
                    "Une erreur technique est survenue "
                    "pendant le traitement de la question."
                )

                st.error(answer)

                with st.expander(
                    "Détail de l'erreur"
                ):
                    st.exception(error)


    # --------------------------------------------------------
    # Historique réponse
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Powered by Mistral AI, FAISS & SQLite | "
    "Hybrid NBA Analyst"
)