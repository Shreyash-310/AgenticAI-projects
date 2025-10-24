Hi can you help me develop the agentic project with the following requirements,
i want to use the google gemini model for my project for embedding and generation part, 

Requirement as follows:
Asked to submit the solution for provided use case, and prepare
    1. a representation outlining their proposed architecture

Use case:  Smart Assistant with Advanced Capabilities 
This use case describes a solution for building an AI powered assistant that enables users to interact with both documents and structured data through conversational interface. The assistant also supports sematic search, contextual memory and reasoning transparency.

Key Features
    1. Document upload and chat interaction
    user can upload document in formats such as pdf, docx, or plain text.
    A chat interface allows users to:
        * Ask questions about the document
        * Summarize specific sections
        * Extract insights or key points
    The assistant should also be able to dynamically access structured data sources (eg. databases) to answer questions beyond the uploaded documents.
    2. Semantic Search
    Users can ask natural language queries, and the assistant will retrieve contextually relevant sections from the documents or structured data.
    The assistant should maintain conversational memory, enabling it to:
        * understand follow up questions
        * maintain context across multiple turns
    3. Tool selection and reasoning
    The assistant should also be capable of tool selection (choosing between document search and structured data or summarization)
    It should explain it's reasoning:
        * Why it chose a particular tool
        * How it derived the response
        * What sources or logic were used

Can use streamlit for UI.