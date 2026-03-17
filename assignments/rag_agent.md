# RAG agent

## Implentation:
A simple RAG agent was created using local documents, embeddings, and a FAISS vectore store. 
The document was split into chunks, embedded with Ollama embeddings, and stored in a retriever. 
A search tool was added so the agent can retrieve relevant document content when answering questions.


## Result:
The agent can now answer questions based on the local document collection instead of relying only on the language model.
This demonstrates a basic Retrieval-Augmented Generation workflow.