import os
import json
from langchain_cohere import ChatCohere
from langchain.schema import HumanMessage
import re

class LLM:
    def __init__(self, model_name: str = "command-r", temperature: float = 0.3, streaming: bool = True):
        """
        Initialize the LLM wrapper for Cohere models.

        Args:
            model_name (str): Name of the model to use, using the default one command-r.
            temperature (float): Sampling temperature for response generation.
            streaming (bool): Whether to enable streaming responses.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.streaming = streaming
        self.api_key = os.getenv("COHERE_API_KEY")


    def _get_client(self):
        """
        Instantiate a Cohere chat client using current settings.

        Returns:
            ChatCohere: A configured LangChain Cohere chat client.
        """
        return ChatCohere(
            cohere_api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming
        )


    def __call__(self, messages: list[HumanMessage], stream: bool = False) -> str:
        """
        Call the LLM with the given messages and return the response.

        Args:
            messages (List[HumanMessage]): List of user messages for context.
            stream (bool): Whether to stream the output.

        Returns:
            str: The generated response.
        """
        client = self._get_client()
        if stream:
            streamed_response = client.stream(messages)
            collected = ""
            for chunk in streamed_response:
                token = chunk.content if hasattr(chunk, "content") else ""
                collected += token
            return collected
        else:
            return client(messages).content


    def generate_answer(self, question: str, context: str, stream: bool = True) -> str:
        """
        Generate an answer based on a question and context.

        Args:
            question (str): The user question.
            context (str): The background context to base the answer on.
            stream (bool): Whether to stream the answer as it generates.

        Returns:
            str: The generated answer.
        """
        prompt = f"""
                Answer the prompt based on the context.\nContext:\n{context}\n\nPrompt: {question}
            """
        try:
            return self([HumanMessage(content=prompt)], stream=False)
        except Exception as e:
            raise Exception(f"[Error calling Cohere: {e}]")


    def generate_subqueries(self, query: str, max_subqueries: int = 15) -> list[str]:
        """
        Use the LLM to split a complex query into simpler sub-queries.

        Args:
            query (str): The complex user query.
            max_subqueries (int): Maximum number of subqueries to return.

        Returns:
            List[str]: A list of simplified subqueries.
        """
        prompt = f"""
        You are a helpful assistant that splits a complex user query into multiple simpler sub-queries.
        Only split the query if it contains multiple distinct questions or requests.
        Don't split into more than {max_subqueries} sub-queries.

        Return your result as a Python list of strings in JSON format.
        Do not include explanations or formatting.

        Example 1:
        Input: "What is LangGraph?"
        Output: ["What is LangGraph?"]

        Example 2:
        Input: "Tell me the revenue, total workers, culture and location of company X"
        Output: ["Tell me the revenue of company X", "Tell me the total workers of company X", "Tell me the culture of company X", "Tell me the location of company X"]

        Now, split this query:
        "{query}"
        """
        try:
            raw_response = self([HumanMessage(content=prompt)])
            cleaned = re.sub(r"```(?:json)?\s*", "", raw_response.strip()) 
            cleaned_response = cleaned.rstrip("`") 
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return parsed[:max_subqueries]
        except Exception:
            print("\nError parsing subqueries response, returning original query.\n")
            pass
        return [query]
