"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import database
import rag
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Qdrant configuration
QDRANT_INFO = {
    "client": QdrantClient(**st.secrets.qdrant_client_credentials),
    "collection_name": "manual-text",
}


def install_playwright_browsers() -> None:
    """Install Playwright browsers for web scraping.

    Checks if browsers are already installed and installs them if needed.
    """
    playwright_dir = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(playwright_dir) or not os.listdir(playwright_dir):
        st.session_state.status.write(":material/comedy_mask: Installing Playwright browsers...")
        try:
            subprocess.run(["playwright", "install"], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install Playwright browsers: {e}")


def get_llm():
    """Initialize the Gemini Pro LLM from Google."""
    st.session_state.status.write(":material/emoji_objects: Setting up Gemini Pro LLM...")
    genai.configure(api_key=st.secrets["google_api_key"])
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        top_p=0.8
    )

def set_google_credentials() -> None:
    """Set Google Cloud credentials for database access.

    Writes credentials from secrets to a temporary file and sets the environment variable.
    """
    st.session_state.status.write(":material/key: Setting Google credentials...")
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


def get_embeddings() -> VertexAIEmbeddings:
    """Initialize the Gemini text embeddings model.

    Returns:
        A VertexAIEmbeddings instance for text-embedding-004.
    """
    st.session_state.status.write(
        ":material/token: Setting up the Gemini text-embedding-004 model..."
    )
    return VertexAIEmbeddings(
        model="text-embedding-004",
        project=st.secrets.google_project_id,
    )


def delete_collection() -> None:
    """Delete the existing Qdrant collection."""
    st.session_state.status.write(":material/delete: Deleting existing Qdrant collection...")
    client = QDRANT_INFO["client"]
    client.delete_collection(collection_name=QDRANT_INFO["collection_name"])


def create_collection() -> None:
    """Create a new Qdrant collection for vector storage."""
    st.session_state.status.write(":material/category: Creating new Qdrant collection...")
    client = QDRANT_INFO["client"]
    client.create_collection(
        collection_name=QDRANT_INFO["collection_name"],
        vectors_config=models.VectorParams(
            size=768,  # Matches text-embedding-004
            distance=models.Distance.COSINE,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=12),
    )


def collection_exists() -> bool:
    """Check if the Qdrant collection exists.

    Returns:
        True if the collection exists, False otherwise.
    """
    client = QDRANT_INFO["client"]
    existing_collections = client.get_collections()
    return any(
        col.name == QDRANT_INFO["collection_name"]
        for col in existing_collections.collections
    )


def points_exist() -> bool:
    """Check if the Qdrant collection contains points.

    Returns:
        True if points exist, False otherwise.
    """
    client = QDRANT_INFO["client"]
    collection_info = client.get_collection(QDRANT_INFO["collection_name"])
    return collection_info.points_count is not None and collection_info.points_count > 0


def get_vector_store(embeddings: VertexAIEmbeddings) -> QdrantVectorStore:
    """Initializes the Qdrant vector store.

    Args:
        embeddings: Embeddings model for vectorization.

    Returns:
        Configured QdrantVectorStore instance.
    """
    st.session_state.status.write(":material/database: Setting up Qdrant vector store...")
    return QdrantVectorStore(
        client=QDRANT_INFO["client"],
        collection_name=QDRANT_INFO["collection_name"],
        embedding=embeddings,
    )


def rebuild_database() -> None:
    """Rebuild the database and vector store from scratch."""
    delete_collection()
    create_collection()
    database.create_images_table()

    docs = database.web_scrape()
    all_splits = database.chunk_text(docs=docs)
    embeddings = get_embeddings()
    st.session_state.vector_store = get_vector_store(embeddings)
    database.index_chunks(
        all_splits=all_splits,
        vector_store=st.session_state.vector_store,
    )


def run_batch_test(test_csv, graph, vector_store):
    """Processes batch test queries and generates results CSV with NDCG, MAP, MRR, and relevance scores, yielding progress updates.

    Args:
        test_csv: Uploaded CSV file with test queries.
        graph: LangGraph instance for query processing.
        vector_store: Qdrant vector store for retrieval.

    Yields:
        Tuple of (current_query, total_queries, results) where results is a list of result dictionaries.
    """
    def dcg_at_k(ranks):
        """Calculate Discounted Cumulative Gain (DCG)."""
        return np.sum([rel / np.log2(rank + 2) for rank, rel in enumerate(ranks)])

    def ndcg_at_k(true_relevance, predicted_scores):
        """Calculate Normalized DCG (NDCG)."""
        if not true_relevance or not predicted_scores:
            return 0.0
        actual_dcg = dcg_at_k([true_relevance[i] for i in np.argsort(predicted_scores)[::-1]])
        ideal_dcg = dcg_at_k(sorted(true_relevance, reverse=True))
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def average_precision(true_relevance, predicted_scores):
        """Calculate Average Precision (AP) for MAP."""
        if not true_relevance or not predicted_scores:
            return 0.0
        sorted_indices = np.argsort(predicted_scores)[::-1]
        relevant = 0
        precision_sum = 0.0
        for i, idx in enumerate(sorted_indices):
            if true_relevance[idx] == 1:
                relevant += 1
                precision_sum += relevant / (i + 1)
        return precision_sum / sum(true_relevance) if sum(true_relevance) > 0 else 0.0

    def reciprocal_rank(true_relevance, predicted_scores):
        """Calculate Reciprocal Rank (RR) for MRR."""
        if not true_relevance or not predicted_scores:
            return 0.0
        sorted_indices = np.argsort(predicted_scores)[::-1]
        for i, idx in enumerate(sorted_indices):
            if true_relevance[idx] == 1:
                return 1.0 / (i + 1)
        return 0.0

    expected_timings = {
        "Vector store retrieval": 0.0,
        "Image fetch": 0.0,
        "Tool execution": 0.0,
        "LLM decision": 0.0,
        "LLM generation": 0.0
    }

    df = pd.read_csv(test_csv)
    grouped = df.groupby(['query_id', 'query_text'])
    total_queries = len(grouped)
    results = []

    for i, ((query_id, query_text), group) in enumerate(grouped):
        ground_truth_ids = set(group['point_id'].astype(str))
        config = {"configurable": {"thread_id": f"test_query_{query_id}"}}
        initial_state = {
            "messages": [
                HumanMessage(content=query_text),
            ],
            "images": [],
            "videos": [],
            "timings": [],
        }
        final_state = list(graph.stream(initial_state, stream_mode="values", config=config))[-1]
        response_message = [
            msg for msg in final_state["messages"]
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ][-1]
        response_text = response_message.content
        tool_message = [msg for msg in final_state["messages"] if msg.type == "tool"]
        retrieved_docs = tool_message[-1].artifact["docs"] if tool_message else []
        timings = final_state["timings"]

        retrieved_results = vector_store.similarity_search_with_relevance_scores(
            query_text, k=len(retrieved_docs) if retrieved_docs else 4
        )
        retrieved_results = sorted(retrieved_results, key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc for doc, _ in retrieved_results]
        predicted_scores = [score for _, score in retrieved_results]
        true_relevance = [
            1 if doc.metadata.get('_id', 'unknown') in ground_truth_ids else 0
            for doc in retrieved_docs
        ]

        ndcg = ndcg_at_k(true_relevance, predicted_scores)
        ap = average_precision(true_relevance, predicted_scores)
        rr = reciprocal_rank(true_relevance, predicted_scores)

        timing_dict = expected_timings.copy()
        for timing in timings:
            if timing["component"] in timing_dict:
                timing_dict[timing["component"]] = timing["time"]

        for doc, score in retrieved_results:
            chunk_id = doc.metadata.get('_id', 'unknown')
            chunk_text = doc.page_content
            chunk_url = doc.metadata.get('source', 'unknown')
            is_in_gt = chunk_id in ground_truth_ids
            results.append({
                'Test query ID': query_id,
                'Query text': query_text,
                'Response text': response_text,
                'Retrieved chunk ID': chunk_id,
                'Retrieved chunk text': chunk_text,
                'Retrieved chunk page URL': chunk_url,
                'Is in ground truth': is_in_gt,
                'Relevance score': score,
                'NDCG': ndcg,
                'MAP': ap,
                'MRR': rr,
                **{f"{component} (s)": time for component, time in timing_dict.items()}
            })

        yield i + 1, total_queries, results
    yield total_queries, total_queries, results


def display_setup() -> None:
    """Renders admin setup controls in the Streamlit UI."""
    with st.expander(label="Admin log-in", expanded=False, icon=":material/lock_open:"):
        password = st.text_input(
            label="Password",
            type="password",
            placeholder="Enter admin password",
        )

        if password == st.secrets.admin_password:
            st.success("Logged in as admin")

            # Database parameters
            st.subheader("Database Parameters")
            db_col1, db_col2, _ = st.columns(3)
            with db_col1:
                new_chunk_size = st.number_input(
                    "Chunk size",
                    min_value=100,
                    max_value=5000,
                    value=st.session_state.get("chunk_size", 1000),
                    step=100,
                )
            with db_col2:
                new_chunk_overlap = st.number_input(
                    "Chunk overlap",
                    min_value=0,
                    max_value=1000,
                    value=st.session_state.get("chunk_overlap", 200),
                    step=50,
                )
            
            if st.button("Update Database"):
                st.session_state.chunk_size = new_chunk_size
                st.session_state.chunk_overlap = new_chunk_overlap
                
                with st.session_state.status:
                    set_google_credentials()
                    embeddings = get_embeddings()
                    rebuild_database()
                    st.session_state.vector_store = get_vector_store(embeddings)
                    st.success(f"Database updated with chunk_size={new_chunk_size}, overlap={new_chunk_overlap}")

            # RAG parameters
            st.subheader("RAG Parameters")
            rag_col1, _, _ = st.columns(3)
            with rag_col1:
                new_k = st.number_input(
                    "Number of chunks to retrieve (k)",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("retrieval_k", 4),
                )
            
            if st.button("Reconfigure RAG"):
                st.session_state.retrieval_k = new_k
                with st.session_state.status:
                    st.session_state.graph = rag.build_graph(
                        llm=st.session_state.llm,
                        vector_store=st.session_state.vector_store,
                        k=st.session_state.retrieval_k,
                    )
                    st.success(f"RAG updated with number of chunks={new_k}")

            # Batch Testing
            st.subheader("Batch Testing")
            test_csv = st.file_uploader(
                "Upload Test CSV",
                type="csv",
                disabled=not st.session_state.vector_store or not st.session_state.graph,
            )
            if test_csv:
                st.session_state.test_csv = test_csv
                if st.button("Run Batch Test", disabled=not st.session_state.test_csv):
                    with st.spinner("Running batch test..."):
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        results = []
                        for current_query, total_queries, batch_results in run_batch_test(
                            st.session_state.test_csv,
                            st.session_state.graph,
                            st.session_state.vector_store,
                        ):
                            results = batch_results
                            progress = current_query / total_queries
                            progress_bar.progress(min(progress, 1.0))
                            progress_text.text(f"Processing query {current_query} of {total_queries}...")

                        st.session_state.results_csv = pd.DataFrame(results).to_csv(index=False)
                        progress_bar.empty()
                        progress_text.empty()
                        st.success("Batch test completed.")
            if st.session_state.results_csv:
                st.download_button(
                    label="Download Results CSV",
                    data=st.session_state.results_csv,
                    file_name="test_results.csv",
                    mime="text/csv"
                )