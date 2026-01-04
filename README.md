# RAG-Powered-Chatbot-to-Turn-Customer-Feedback-into-Actionable-Insights

A Retrieval-Augmented Generation (RAG) chatbot that analyzes real financial customer complaints to deliver evidence-based insights. It uses semantic search with vector databases and LLMs to help product, support, and compliance teams quickly identify trends across financial services.

Task 1: Data Understanding and Preprocessing Summary
Overview of the Data For the first part of the project, we looked at the Consumer Financial Protection Bureau (CFPB) dataset to find complaints that matter most to CrediTrust. Since the full file is very large, we started with a sample of 100,000 rows. After filtering for our five specific products (Credit Cards, Personal Loans, Savings Accounts, and Money Transfers) and removing any empty stories, we were left with a high-quality set of 205 complaints to begin our analysis.

Key Findings from EDA Our initial analysis shows that "Checking or savings account" is currently the biggest source of customer feedback, making up about 80% of our filtered sample. We also looked at how much customers like to write; the average complaint is quite detailed, which is perfect for our AI model to learn from later. Most stories focus on specific issues like account management and transfer delays.

Cleaning for the AI To make sure our chatbot gives the best answers, we cleaned the text narratives. We made everything lowercase and removed special symbols and numbers. This "polishing" step ensures that the AI focuses only on the important words when it tries to understand what Asha's customers are worried about. We saved this cleaned data as filtered_complaints.csv so it is ready for the next step.
