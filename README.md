# Pharma Knowledge Assistant

## Overview
The **Pharma Knowledge Assistant** is an intelligent application designed to enhance access to pharmaceutical knowledge using cutting-edge AI technologies. Built with GPT-2 from Huggingface and an interactive user interface powered by Streamlit, this project incorporates Retrieval-Augmented Generation (RAG) and an Agentic framework to deliver dynamic and accurate responses to user queries. It provides features such as question answering, recommendations, alternatives generation, and summarization tailored for pharmaceutical datasets.

---

## Features
1. **Question Answering**
   - Responds to natural language queries about pharmaceutical products.
   - Example: *"What is the composition and primary use of Paracetamol?"*

2. **Recommendations**
   - Provides tailored advice or warnings based on user inputs.
   - Example: *"Can I take Ibuprofen if I have a history of stomach ulcers?"*

3. **Alternatives Generation**
   - Suggests product alternatives when specific medications are not suitable.

4. **Summarization**
   - Delivers concise summaries of pharmaceutical product details.

5. **Agentic Framework**
   - Integrates all features into a graph-based Agentic framework, where each functionality operates as a node.

6. **Search Integration**
   - Leverages external search tools when the dataset lacks sufficient information.

7. **GUI Chatbot**
   - A user-friendly chatbot interface built with Streamlit, enabling real-time interactions.

---

## Technologies Used
- **Large Language Model**: [GPT-2](https://huggingface.co/gpt2) via Huggingface.
- **Frameworks**: Streamlit, LangChain, LangGraph.
- **Backend**: Python-based architecture leveraging Agentic RAG principles.
- **Evaluation Tools**: Metrics evaluation using TruLens for relevance, groundedness, and comprehensiveness.

---

## Architecture
The application is structured as a graph-based Agentic system:
- **Nodes**:
  - Question Answering using RAG.
  - Recommender system.
  - Alternatives generator.
  - Summarizer.
- **Router Nodes**:
  - Routes user queries to the appropriate node.
  - Handles fallback mechanisms for web search when required.
- **Context Sharing**:
  - Maintains a unified context across nodes for seamless interactions.

---

## Installation and Usage
### Prerequisites
- Python 3.9 or above
- GPU (optional for improved performance)
- Streamlit installed

### Steps
1. Clone the repository:
```bash      
git clone https://github.com/your-username/pharma-knowledge-assistant.git
cd pharma-knowledge-assistant
```
   
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start the application
```bash
streamlit run streamlit.py
```
Interact with the chatbot via the web-based GUI

The following metrics are used to evaluate the system:

Context Relevance: Ensuring the response aligns with the query context.
Answer Relevance: Accuracy and utility of responses.
Groundedness: Validity of responses backed by the dataset.
Comprehensiveness: Coverage of the query requirements.

### Acknowledgments
* PES University ESA Hackathon for the foundational problem statement.
* Huggingface, Streamlit, LangChain, and LangGraph communities for their resources and support.

