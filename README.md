# RAG-Powered-Chatbot-to-Turn-Customer-Feedback-into-Actionable-Insights

Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to analyze real financial customer complaints and transform unstructured feedback into evidence-based insights. The system enables product, support, and compliance teams to quickly identify recurring issues across multiple financial services.

By combining semantic search with large language models (LLMs), the chatbot retrieves relevant complaint narratives and generates concise, grounded answers to user questions.

Data Understanding and Preprocessing (Task 1)
Dataset Overview

The project uses complaint data from the Consumer Financial Protection Bureau (CFPB). Due to the large size of the original dataset, an initial sample of 100,000 records was used. After filtering for the target products—Credit Cards, Personal Loans, Savings Accounts, and Money Transfers—and removing records with missing narratives, a clean and high-quality subset of 205 complaints was obtained for exploratory analysis.

Key EDA Findings

Complaints related to checking and savings accounts dominate the dataset, accounting for approximately 80% of the filtered sample.

Most complaint narratives are detailed and descriptive, making them well-suited for downstream NLP tasks.

Common themes include account management issues and transfer delays.

Text Cleaning

To improve embedding quality, complaint narratives were normalized by:

Converting text to lowercase

Removing special characters and numeric symbols

The cleaned dataset was saved as filtered_complaints.csv for use in subsequent stages of the pipeline.

Vector Store Indexing and Embedding (Task 2)
Sampling Strategy

A stratified sample of 10,000 complaints was created to ensure balanced representation across all target product categories. This approach prevents model bias toward more frequent complaint types.

Text Chunking

Complaint narratives were divided into overlapping text chunks:

Chunk size: 500 characters

Overlap: 50 characters

This strategy preserves contextual continuity and ensures important information is not lost during segmentation.

Embeddings and Vector Store

The all-MiniLM-L6-v2 embedding model was used to convert text chunks into dense vector representations that capture semantic meaning. These embeddings were indexed using FAISS (Facebook AI Similarity Search) and persisted in the vector_store/ directory.

This vector store serves as the core retrieval engine, enabling fast and accurate semantic search across thousands of complaint narratives.

Key Technologies

Python

FAISS

Sentence Transformers (all-MiniLM-L6-v2)

Retrieval-Augmented Generation (RAG)

Project Goal

The final system allows internal stakeholders to ask natural-language questions about customer complaints and receive reliable, context-backed insights—shifting analysis from manual review to intel
