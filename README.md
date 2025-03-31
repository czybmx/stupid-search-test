# stupid-search-test
This Python script implements an AI-powered question-answering system that processes user queries, retrieves relevant information from the web, and generates detailed responses. It integrates multiple technologies, including natural language processing (NLP), vector search, and external APIs, to provide accurate and context-aware answers.

# AI-Powered Search and Answer System

This project is an AI-powered system that processes user queries, performs web searches, and generates detailed answers using a combination of local models, external APIs, and advanced natural language processing techniques. It leverages the Tavily API for search, FAISS for vector search, and various language models (via Ollama) for reasoning and response generation.

## Features
- **Dynamic Identity & Expertise Assignment**: Determines the best persona and skills to answer queries.
- **Web Search Integration**: Uses the Tavily API to fetch real-time information.
- **Vector Search**: Employs FAISS for efficient similarity search over retrieved documents.
- **Modular Design**: Combines multiple AI models for intent analysis, exploration, and answer generation.
- **Error Handling**: Robust logging and fallback mechanisms for API or model failures.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Operating System**: Compatible with Windows, macOS, or Linux.
- **Hardware**: GPU recommended for faster model inference (CUDA support required for GPU usage).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install requests torch transformers faiss-cpu numpy python-dotenv aiohttp
   ```
   For GPU support, replace `faiss-cpu` with `faiss-gpu` (requires CUDA).

4. **Install Ollama**:
   - Download and install Ollama from [ollama.ai](https://ollama.ai/).
   - Ensure the Ollama server is running locally (`ollama serve`).
   - Pull required models:
     ```bash
     deepseek-r1:14b 
     gemma:12b [But im using google_gemma-3-12b-it-Q6_K_L:latest (from Huggingface GGUF)]
     ```

5. **Set Up Environment Variables**:
   - Create a `.env` file in the project root:
     ```bash
     echo "TAVILY_API_KEY=your-tavily-api-key" > .env
     ```
   - Replace `your-tavily-api-key` with your actual Tavily API key (sign up at [tavily.com](https://tavily.com) to get one).

6. **Download the Embedding Model**:
   - The script automatically downloads the `BAAI/bge-m3` model on first run if not present. Ensure you have an internet connection and sufficient disk space (~2GB).

## Tutorial

### Step 1: Prepare the Environment
After installation, verify that the Ollama server is running:
```bash
ollama serve
```
Check that your `.env` file contains a valid Tavily API key.

### Step 2: Run the Script
Execute the script with:
```bash
python main.py
```
You’ll be prompted to enter a query:
```
User: What are the latest natural disasters in 2025?
```

### Step 3: Understand the Workflow
1. **Input Processing**: The script analyzes your query to determine intent and required expertise.
2. **Search**: It generates search queries and fetches results via Tavily API.
3. **Vector Indexing**: Retrieved documents are encoded and indexed using FAISS.
4. **Answer Generation**: Multiple AI models collaborate to explore the topic and provide a final answer.

### Step 4: View Output
The final answer is printed to the console, prefixed with `===== FINAL ANSWER =====`.

## Usage Methods

### Basic Usage
Run the script and input a question:
```bash
python main.py
User: How does climate change affect agriculture?
```

### Debugging
Enable detailed logging by modifying the script’s logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```
This provides insights into each step (e.g., API calls, model responses).

### Customizing Models
Edit the script to use different Ollama models:
- Change `deepseek-r1:14b` or `google_gemma-3-12b-it-Q6_K_L:latest` in the `call_ollama` function calls.
- Ensure the new models are pulled via `ollama pull <model-name>`.

### Running on GPU
If you have a CUDA-enabled GPU:
1. Install `faiss-gpu` and ensure PyTorch is built with CUDA support.
2. The script auto-detects and uses the GPU if available.

## Troubleshooting
- **Model Download Fails**: Check internet connectivity and disk space in the `models/` directory.
- **Ollama Errors**: Ensure the Ollama server is running (`ollama serve`) and models are pulled.
- **Tavily API Issues**: Verify your API key in `.env` and ensure you have an active subscription.
- **Memory Issues**: Reduce `max_results` in `tavily_search` or use a machine with more RAM.

## Contributing
Feel free to submit pull requests or open issues on GitHub. Contributions to improve performance, add features, or enhance documentation are welcome!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
