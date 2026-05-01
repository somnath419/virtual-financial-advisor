import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env (no-op if already set)
load_dotenv()


def get_llm() -> ChatGroq:
    """
    Initialize and return a configured Groq language model.

    Reads the Groq API key from the GROQ_API_KEY environment variable
    (set via .env or the shell) and creates a ChatGroq instance configured
    for financial advice generation.

    The model defaults to ``llama-3.3-70b-versatile`` (free tier) and can
    be overridden by setting the ``GROQ_MODEL`` environment variable,
    e.g. ``GROQ_MODEL=llama-3.1-8b-instant`` for a faster response.

    Free models available on Groq:
        - llama-3.3-70b-versatile   (default – best quality)
        - llama-3.1-8b-instant      (fastest)
        - mixtral-8x7b-32768        (large context window)
        - gemma2-9b-it              (Google Gemma 2)

    Returns:
        ChatGroq: Configured language model with temperature=0.2
                  for consistent, focused responses.

    Raises:
        ValueError: If the GROQ_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and add it to .env"
        )

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model,
        temperature=0.2,
        api_key=api_key,
    )


def generate_advice(user_context: str) -> str:
    """
    Generate concise financial advice based on a user context string.

    Builds a LangChain pipeline (prompt → LLM → string parser) and invokes
    it with the supplied context.  The system prompt instructs the model to
    act as a professional financial advisor and focus on actionable steps.

    Args:
        user_context (str): A description of the user's financial situation
                            or a specific question requiring advice.

    Returns:
        str: Concise, actionable financial advice tailored to the context.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional virtual financial advisor. "
            "Be concise and practical. "
            "Analyze the context and provide actionable steps. "
            "Avoid disclaimers.",
        ),
        ("human", "{context}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": user_context})