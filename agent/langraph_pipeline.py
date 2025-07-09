from langgraph.graph import StateGraph, START, END
from agent.tavily_search import tavily_search
from agent.llm_response import LLM
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from typing import TypedDict
from util import pretty_print

class AgentState(TypedDict):
    query: str
    subqueries: list[str]
    combined_context: str
    response: str
    revised_response: str


class TavilyRAGPipeline:
    """A LangGraph pipeline for query decomposition, Tavily search, and LLM response generation."""

    def __init__(self, show_subqueries: bool = False):
        """
        Initialize the pipeline.

        Args:
            show_subqueries (bool): Whether to print subqueries during execution.
        """
        self.show_subqueries = show_subqueries
        self.llm = LLM()
        self.graph = self._build_graph()

    # Nodes
    def _query_parser_node(self, state: AgentState) -> dict:
        """
        Decompose the input query into subqueries.

        Args:
            state (AgentState): The current agent state containing the user query.

        Returns:
            Dict[str, Any]: Dictionary with generated subqueries.
        """
        query = state["query"]
        pretty_print(query, subtext="Human Query", color="95")
        subqueries = self.llm.generate_subqueries(query)
        return {"subqueries": subqueries}

    async def _search_and_context_node(self, state: AgentState) -> dict:
        """
        Fetch search results for each subquery and combine them into a context string.

        Args:
            state (AgentState): Agent state with subqueries.

        Returns:
            Dict[str, str]: Dictionary with the combined search context.
        """
        snippets = []
        for subquery in state["subqueries"]:
            if self.show_subqueries and len(state["subqueries"]) > 1:
                pretty_print(subquery, subtext="Sub Query", color="92")
            use_advanced = len(subquery.split()) > 8 or len(state["subqueries"]) > 3
            depth = "advanced" if use_advanced else "basic"
            context = await tavily_search(subquery, search_depth=depth)
            snippets.append(context)
        combined_context = "\n\n".join(snippets)
        return {"combined_context": combined_context}

    def _llm_node(self, state: AgentState) -> dict:
        """
        Generate the final answer using the LLM and combined context.

        Args:
            state (AgentState): Agent state with query and combined context.

        Returns:
            Dict[str, str]: Dictionary with the generated LLM response.
        """
        response = self.llm.generate_answer(
            question=state["query"],
            context=state["combined_context"],
            stream=False,
        )
        response = "".join(chunk for chunk in response)
        return {"response": response}

    def _critique_and_revise_node(self, state: AgentState) -> dict:
        critique_prompt = f"""
        You are a helpful assistant tasked with reviewing and improving an AI-generated response.
        Here is the original query:
        {state['query']}

        Here is the context provided for answering:
        {state['combined_context']}

        And here is the response to review:
        {state['response']}

        Please check the response for accuracy, clarity, and completeness. If the original response is already excellent, return the word ok.
        Otherwise, return the word inaccurate.
        """
        revised_response = self.llm.generate_answer(
            question=critique_prompt,
            context=state["combined_context"],
            stream=False,
        )
        return {"revised_response": revised_response}
    

    def _publish_node(self, state: AgentState) -> dict:
        pretty_print("", subtext="AI Response", color="96")
        self._print_wrapped_under_bar(state['response'], 80)
        # Print any remaining buffer
        if hasattr(self, "_char_buffer") and self._char_buffer.strip():
            print(self._char_buffer.strip() + "\n")
            del self._char_buffer
        return {"response": state["response"]}


    def _critique_condition(self, state: AgentState) -> str:
        return "search_context" if "inaccurate" in state["response"].lower() else "publish"


    def _build_graph(self):
        """
        Define the LangGraph pipeline with nodes and edges.

        Returns:
            StateGraph: The compiled LangGraph pipeline.
        """
        builder = StateGraph(AgentState)
        #Using RAM memory instead of disk for simplicity and efficiency, works good because the project runs locally
        memory = MemorySaver()

        builder.add_node("parse", self._query_parser_node)
        builder.add_node("search_context", self._search_and_context_node, is_async=True)
        builder.add_node("llm", self._llm_node)
        builder.add_node("critique_and_revise", self._critique_and_revise_node)
        builder.add_node("publish", self._publish_node)

        builder.add_edge(START, "parse")
        builder.add_edge("parse", "search_context")
        builder.add_edge("search_context", "llm")
        builder.add_edge("llm", "critique_and_revise")
        builder.add_edge("critique_and_revise", "publish")
        builder.add_edge("publish", END)

        
        builder.add_conditional_edges(
            "critique_and_revise",
            self._critique_condition,
            {
                "search_context": "search_context",
                "publish": "publish"
            }
        )

        # Add checkpointing to avoid redoing search or LLM steps
        return builder.compile(checkpointer=memory)
    

    def visualize(self):
        """
        Visualize the LangGraph pipeline structure using Mermaid diagram.
        """
        print("LangGraph Structure:")
        img = self.graph.get_graph().draw_mermaid_png()
        display(Image(img))


    def _print_wrapped_under_bar(self, text: str, max_width: int = 80):
        """
        Wraps and prints text to stdout when line length exceeds max_width.

        Args:
            text (str): Text to print in a wrapped format.
            max_width (int): Maximum characters per line before wrapping.
        """
        if not hasattr(self, "_char_buffer"):
            self._char_buffer = ""
        self._char_buffer += text
        while len(self._char_buffer) >= max_width:
            break_idx = self._char_buffer.rfind(" ", 0, max_width)
            if break_idx == -1:
                break_idx = max_width
            line = self._char_buffer[:break_idx].rstrip()
            print(line)
            self._char_buffer = self._char_buffer[break_idx:].lstrip()


    def get_graph(self):
        return self.graph
