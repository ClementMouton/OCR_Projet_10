# NBA Analyst AI — Assistant RAG hybride avec Mistral

Projet réalisé dans le cadre du parcours **Data Scientist / Machine Learning d'OpenClassrooms**.

L'objectif est de construire un assistant conversationnel capable d'exploiter plusieurs types de données NBA :

- des contenus documentaires non structurés, notamment des discussions Reddit ;
- des statistiques individuelles de joueurs sous forme tabulaire ;
- des questions nécessitant de combiner informations documentaires et statistiques.

Le système repose sur une architecture hybride associant **RAG, génération SQL et routage automatique des questions**.

---

## 1. Objectif du projet

Un RAG classique est adapté aux questions portant sur du contenu textuel, mais moins pertinent lorsqu'une réponse nécessite un calcul ou un filtrage précis sur des données structurées.

Le projet sépare donc deux sources d'information :

- **Données non structurées** : extraction/OCR, chunking, embeddings Mistral, indexation FAISS et recherche sémantique.
- **Données structurées** : statistiques NBA intégrées dans SQLite et interrogées via un SQL Tool.

Un routeur choisit automatiquement le pipeline adapté à chaque question.

---

## 2. Architecture générale

```text
                         Question utilisateur
                                  |
                                  v
                         +------------------+
                         | Routeur Mistral  |
                         +------------------+
                                  |
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
            RAG                  SQL                HYBRID
             |                    |                    |
             v                    v             +------+------+
      Recherche FAISS       SQL Tool            |             |
             |                    |             v             v
             v                    v           FAISS         SQLite
     Chunks documentaires     SQLite             |             |
             |                    |             +------+------+
             +--------------------+--------------------+
                                  |
                                  v
                              Mistral
                                  |
                                  v
                         Réponse utilisateur
```

Une quatrième route, `OUT_OF_SCOPE`, identifie les questions ne relevant pas du périmètre NBA couvert.

---

## 3. Routes disponibles

### `rag`

Questions dont la réponse doit être recherchée dans les documents textuels.

> Quelles équipes ont impressionné les commentateurs pendant les playoffs ?

### `sql`

Questions nécessitant une interrogation précise des statistiques NBA.

> Qui est le joueur le plus précis à trois points parmi ceux qui en ont tenté au moins 100 ?

### `hybrid`

Questions nécessitant simultanément des informations documentaires et des statistiques structurées.

> Quels joueurs des Minnesota Timberwolves sont mis en avant par les commentateurs et que montrent leurs statistiques individuelles disponibles ?

### `out_of_scope`

Questions ne correspondant pas au périmètre couvert par le système.

---

## 4. Sources de données

### Documents Reddit

Les discussions Reddit constituent la principale source documentaire du RAG.

Lorsque le texte d'un PDF n'est pas suffisamment exploitable directement, le pipeline utilise **EasyOCR**. Le texte obtenu est ensuite nettoyé, découpé et indexé.

### Statistiques NBA

Les statistiques individuelles sont fournies dans :

```text
inputs/regular NBA.xlsx
```

Le script `load_excel_to_db.py` valide les données puis les insère dans :

```text
database/nba.db
```

La table principale est `player_stats`.

---

## 5. Pipeline d'indexation documentaire

```bash
python indexer.py
```

```text
PDF
 |
 v
Extraction texte
 |
 +---- texte insuffisant ----> OCR EasyOCR
 |
 v
Nettoyage
 |
 v
Chunking
 |
 v
Embeddings Mistral
 |
 v
Index FAISS
```

Configuration actuelle :

```text
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
SEARCH_K = 5
```

Artefacts :

```text
vector_db/
├── faiss_index.idx
└── document_chunks.pkl
```

L'index actuellement utilisé contient **100 vecteurs pour 100 chunks documentaires**.

---

## 6. Base SQLite et SQL Tool

```bash
python load_excel_to_db.py
```

Le script lit le fichier Excel, sélectionne les colonnes utiles, valide les données, crée SQLite et alimente `player_stats`.

Exemple de requête générée :

```sql
SELECT
    player,
    team,
    three_pa,
    three_p_pct
FROM player_stats
WHERE three_pa >= 100
ORDER BY three_p_pct DESC
LIMIT 1;
```

---

## 7. Sécurisation des requêtes SQL

Une requête générée par le LLM n'est pas exécutée directement.

La validation contrôle notamment :

- le caractère autorisé de la lecture ;
- l'absence de modification des données ;
- la présence d'une seule instruction ;
- l'utilisation exclusive des tables autorisées.

Les opérations `INSERT`, `UPDATE`, `DELETE`, `DROP` ou `PRAGMA` sont notamment rejetées.

---

## 8. Modèles utilisés

### Génération

```text
mistral-small-latest
```

Utilisé pour le routage, la génération SQL et la génération des réponses.

### Embeddings

```text
mistral-embed
```

Utilisé pour la représentation vectorielle des chunks et des questions.

---

## 9. Installation

```bash
git clone https://github.com/ClementMouton/OCR_Projet_10.git
cd OCR_Projet_10
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```

Puis :

```bash
pip install -r requirements.txt
```

Créer un `.env` à partir de `.env.example` et renseigner au minimum :

```text
MISTRAL_API_KEY=VOTRE_CLE_API
```

---

## 10. Préparation des données

Créer la base SQLite :

```bash
python load_excel_to_db.py
```

Construire l'index documentaire :

```bash
python indexer.py
```

L'OCR peut prendre plusieurs minutes sur CPU.

---

## 11. Lancer l'application

```bash
streamlit run MistralChat.py
```

Le pipeline sélectionne automatiquement la route adaptée à la question.

---

## 12. Évaluation

### Tests unitaires

```bash
python -m pytest -v
```

Résultat actuel :

```text
21 passed
```

Les tests couvrent :

- l'intégrité SQLite ;
- les résultats statistiques de référence ;
- la sécurité du SQL Tool ;
- l'intégrité FAISS ;
- la cohérence vecteurs/chunks.

### Routeur

Le benchmark couvre les quatre routes.

```text
12 / 12 routes correctement identifiées
Accuracy : 100 %
```

Répartition :

```text
RAG          : Q01 à Q05
OUT_OF_SCOPE : Q06
SQL          : Q07 à Q10
HYBRID       : Q11 à Q12
```

### SQL

```text
4 / 4 requêtes correctes
Exactitude SQL : 100 %
```

---

## 13. Évaluation RAGAS

RAGAS évalue les routes `rag` et `hybrid`. Les routes `sql` sont évaluées séparément et `out_of_scope` est exclue.

| Métrique | Score |
|---|---:|
| Faithfulness | 0,7000 |
| Context Precision | 0,6127 |
| Context Recall | 0,5476 |

### Interprétation

**Faithfulness — 0,7000** : les réponses restent globalement fondées sur le contexte fourni.

**Context Precision — 0,6127** : une majorité des éléments remontés sont utiles, mais certains chunks restent peu pertinents.

**Context Recall — 0,5476** : une partie significative des informations nécessaires est retrouvée, mais le retrieval peut encore être amélioré.

### Response Relevancy

`ResponseRelevancy` a été testée mais retournait des valeurs `NaN` avec l'intégration Mistral/RAGAS utilisée dans l'environnement du projet. Elle a donc été retirée de l'évaluation finale.

---

## 14. Stratégie de validation

```text
Tests pytest
    |
    +--> intégrité SQLite
    +--> sécurité SQL
    +--> intégrité FAISS

Benchmark routeur
    |
    +--> RAG / SQL / HYBRID / OUT_OF_SCOPE

Évaluation SQL
    |
    +--> exactitude déterministe

RAGAS
    |
    +--> qualité du retrieval
    +--> fidélité au contexte
```

---

## 15. Structure du projet

```text
OCR_Projet_10/
│
├── MistralChat.py
├── indexer.py
├── load_excel_to_db.py
├── evaluate_hybrid.py
├── evaluate_ragas.py
├── test_router.py
├── test_sql_tool.py
│
├── inputs/
├── database/
│   └── nba.db
├── vector_db/
│   ├── faiss_index.idx
│   └── document_chunks.pkl
├── evaluation/
│   ├── test_cases.json
│   ├── run_ragas.py
│   ├── run_ragas_hybrid.py
│   ├── evaluate_sql.py
│   └── results/
├── tests/
│   ├── test_sql_database.py
│   ├── test_sql_security.py
│   └── test_vector_store.py
├── utils/
│   ├── config.py
│   ├── rag_pipeline.py
│   ├── sql_tool.py
│   └── vector_store.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 16. Limites actuelles

- Le retrieval documentaire peut encore être amélioré, notamment au regard du Context Recall.
- EasyOCR est relativement lent sur CPU.
- Le système dépend de la disponibilité, de la latence et des quotas de l'API Mistral.
- La génération SQL reste dépendante de l'interprétation de la question par le LLM, malgré la validation de sécurité.
- Le benchmark d'évaluation reste limité à un nombre restreint de questions.

---

## 17. Améliorations possibles

- enrichir le jeu d'évaluation ;
- optimiser le chunking et le retrieval ;
- ajouter un reranker ;
- adapter dynamiquement `SEARCH_K` ;
- mesurer la latence par route ;
- suivre les coûts API ;
- ajouter du monitoring ;
- mettre en cache certaines requêtes ;
- renforcer la gestion des indisponibilités API.

---

## 18. Technologies utilisées

- Python
- Mistral AI
- FAISS
- SQLite
- Streamlit
- EasyOCR
- PyMuPDF
- pandas
- Pydantic
- RAGAS
- pytest
- LangChain / intégrations Mistral

---

## Auteur

Projet réalisé par **Clément Mouton** dans le cadre du parcours OpenClassrooms Data Scientist / Machine Learning.
