Create a virtual environment

```
python -m venv venv
```

Activate the virtual environment

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

Install libraries

```
pip install -r requirements.txt
```

Create accounts

- Create a free account on Pinecone: https://www.pinecone.io/
- Create an API key for OpenAI: https://platform.openai.com/api-keys

Add API keys to .env file

- Rename .env.example to .env
- Add the API keys for Pinecone and OpenAI to the .env file

<h3>Executing the scripts</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```
python sample_ingestion.py
python sample_retrieval.py
python ingestion.py
python retrieval.py
streamlit run chatbot_rag.py
```
