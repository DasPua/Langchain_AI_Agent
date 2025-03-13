# LangChain AI Agent

LangChain AI Agent is a Python-based intelligent agent that leverages the LangChain framework and OpenAI models to interact with users, perform tasks, and provide contextual responses.

## Features

- Natural language understanding and generation.
- Task automation with LangChain.
- Integration with OpenAI's powerful LLMs.
- Uses external tools like Tavily and HuggingFace APIs.

## Prerequisites

Make sure you have the following installed:

- Python 3.9 or higher
- pip (Python package installer)
- API Keys for:
  - OpenAI
  - Tavily
  - HuggingFace

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/DasPua/Langchain_AI_Agent.git
   cd Langchain_AI_Agent
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install the required dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**  
   Create a `.env` file in the root directory and add your API keys like this:  
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ```

## Usage

To run the AI agent, execute the following command in the project directory:

```bash
python exe.py
```

## License

This project is licensed under the MIT License.

---
