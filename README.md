# Implementing Adaptive RAG: A Step Towards More Intelligent and Flexible AI with LangGraph

In the rapidly evolving landscape of AI and natural language processing, we're constantly seeking ways to improve the reliability, accuracy, and adaptability of our systems. Today, we're exploring an exciting advancement: Adaptive RAG (Retrieval-Augmented Generation). This implementation, built using LangGraph, takes us a step closer to AI systems that can not only self-reflect and self-improve but also adaptively choose the most appropriate information source based on the input query.

## What is Adaptive RAG?

Adaptive RAG is an extension of the traditional RAG (Retrieval-Augmented Generation) system. While RAG enhances language model outputs by retrieving relevant information from a knowledge base, Adaptive RAG goes further by introducing:

1. Query routing to choose between local vectorstore and web search
2. Self-evaluation of retrieved documents and generated responses
3. Multiple attempts at generation if initial results are unsatisfactory
4. Ability to seek additional information when needed

## Components and Workflow

Our Adaptive RAG implementation consists of the following key components:

1. **Query Router**: Decides whether to use the local vectorstore or perform a web search based on the input question.
2. **Retriever**: Fetches relevant documents from the chosen source (vectorstore or web).
3. **Document Grader**: Evaluates the relevance of retrieved documents.
4. **Generator**: Produces a response using the retrieved information.
5. **Generation Grader**: Assesses the generated response for hallucinations and relevance.

The workflow is defined using a `StateGraph` from LangGraph, allowing for complex, conditional paths through the system.

## Code Breakdown

Let's examine the key parts of our Adaptive RAG implementation:

### Graph Definition (graph.py)

```python
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVER, retriever_node)
workflow.add_node(GRADE_DOCS, grading_node)
workflow.add_node(GENERATE, generate_node)
workflow.add_node(WEBSEARCH, websearch_node)

workflow.set_conditional_entry_point(
    query_router_conditional_edge, {VECTORSTORE: RETRIEVER, WEBSEARCH: WEBSEARCH}
)

workflow.add_edge(RETRIEVER, GRADE_DOCS)
workflow.add_conditional_edges(
    GRADE_DOCS,
    grade_conditional_node,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {NOT_SUPPORTED: GENERATE, NOT_USEFUL: WEBSEARCH, USEFUL: END},
)
```

This graph structure allows for adaptive routing and multiple paths through the system, a key aspect of Adaptive RAG.

### Query Routing (router.py)

```python
def query_router_conditional_edge(state: GraphState) -> str:
    question = state["question"]
    query_router_result = query_router_prompt_chain.invoke(input={"question": question})
    destination = query_router_result.destination
    print("Query router destination is {}".format(destination))
    return destination
```

This function demonstrates the adaptive nature of the system by routing queries to either the local vectorstore or web search based on the content of the question.

### State Management (state.py)

```python
class GraphState(TypedDict):
    question: str
    generation: str
    websearch: bool
    documents: List[str]

class QueryRouter(BaseModel):
    destination: Literal["VECTORSTORE", "WEBSEARCH"] = Field(
        description="Given a question choose to route it to vectorstore or websearch"
    )
```

These classes define the state of the graph and the structure for the query router's decision, enabling the adaptive behavior of the system.

## Key Benefits of Adaptive RAG

1. **Improved Accuracy**: By evaluating its own outputs and choosing the most appropriate information source, Adaptive RAG can reduce hallucinations and irrelevant responses.

2. **Flexibility**: The system can adapt to different types of queries, using local knowledge when appropriate and falling back to web search when needed.

3. **Iterative Improvement**: Through multiple generation attempts, the system can refine its responses.

4. **Transparency**: The self-evaluation process provides insight into the system's decision-making.

## Test Scenarios

To demonstrate the capabilities of our Adaptive RAG system, we can consider the following test scenarios:

1. **Local Knowledge Query**: 
   Input: "What are common Java interview questions?"
   Expected Behavior: The query router should direct this to the VECTORSTORE, as it's within the system's local knowledge.

2. **Web Search Query**:
   Input: "What are the latest developments in quantum computing?"
   Expected Behavior: The query router should direct this to WEBSEARCH, as it's likely outside the system's local knowledge.

3. **Ambiguous Query**:
   Input: "How do I prepare a risotto?"
   Expected Behavior: This could go either way. If cooking recipes are in the vectorstore, it should go there. Otherwise, it should route to WEBSEARCH.

4. **Iterative Improvement**:
   Input: "Explain the concept of polymorphism in object-oriented programming."
   Expected Behavior: The system might first retrieve information from the vectorstore. If the generation is graded as NOT_SUPPORTED or NOT_USEFUL, it should attempt to regenerate or perform a web search for more information.

To run these tests, you can use the provided main block in graph.py:

```python
if __name__ == "__main__":
    res = graph.invoke(input={"question": "how to calculate XIRR in mutual funds?"})
    print(res)
```

Replace the question with each of the test scenarios to observe the system's behavior.

## Conclusion

Adaptive RAG represents a significant advancement in our quest for more reliable, accurate, and flexible AI systems. By incorporating query routing, self-reflection, and self-improvement mechanisms, we're moving closer to AI that can critically evaluate its own outputs, adapt its approach when needed, and choose the most appropriate information sources.

This implementation, built with LangGraph, demonstrates the power of flexible, modular design in creating sophisticated AI workflows. As we continue to refine and expand on these concepts, we open up exciting possibilities for AI systems that are not just more accurate, but also more adaptable and trustworthy.

The journey towards more intelligent and flexible AI is ongoing, and Adaptive RAG is a promising step in that direction. It challenges us to think not just about how AI can process information, but how it can question its own processes and adapt to different types of queries and information needs.