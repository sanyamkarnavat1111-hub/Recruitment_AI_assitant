from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Literal

# Define chat_state TypedDict to structure the state
class chat_state(TypedDict):
    message: Literal['yes', 'no']
    output: str

# Define the function to classify the route
def classify_route(state: chat_state) -> Literal['next', 'stop']:
    if state['message'] == 'yes':
        return 'next'
    else:
        return 'stop'

# Define next_node state transformation function
def next_node(state: chat_state) -> chat_state:
    return {
        "output": "Next Node"
    }

# Define stop_node state transformation function
def stop_node(state: chat_state) -> chat_state:
    return {
        "output": "Stop Node"
    }

# Create a StateGraph with the chat_state schema
graph = StateGraph(state_schema=chat_state)

# Add nodes to the graph with their respective functions
graph.add_node("next_node", next_node)
graph.add_node("stop_node", stop_node)

# Add edges between nodes - START directly connects to conditional edges
graph.add_conditional_edges(
    START,
    classify_route,
    {
        "next": "next_node",
        "stop": "stop_node"
    }
)

# End the workflow after either next_node or stop_node
graph.add_edge("next_node", END)
graph.add_edge("stop_node", END)

# Compile the workflow
workflow = graph.compile()

if __name__ == "__main__":
    # Test with 'yes' input
    input_state = {
        "message": "yes"
    }
    
    output = workflow.invoke(input=input_state)
    print("Output for 'yes':", output)
    
    # Test with 'no' input
    input_state_no = {
        "message": "no"
    }
    
    output_no = workflow.invoke(input=input_state_no)
    print("Output for 'no':", output_no)