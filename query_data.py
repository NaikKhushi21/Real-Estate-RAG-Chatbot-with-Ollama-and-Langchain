import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Enhanced, domain-specific prompt template for real estate queries.
PROMPT_TEMPLATE = """
You are an expert Real Estate Analyst with deep knowledge in property valuations, market trends, zoning regulations, and financial projections.
Use the following context to answer the real estate question accurately.

Question: {question}

Context:
{context}

Provide a detailed, insightful answer focusing on investment opportunities, regulatory insights, or financial analysis as applicable.
"""

def query_rag(query_text: str):
    # Prepare the vector store using the embedding function.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform a similarity search to retrieve the top relevant document chunks.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt using the domain-specific template.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Invoke the language model to generate the answer.
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    
    # Optionally, print out the source document IDs for reference.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

if __name__ == "__main__":
    main()
