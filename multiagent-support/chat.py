from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated, Literal
import random
import re

# Define chatbot state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str
    ticket_number: str
    customer_name: str

# Agent Specifications
CLASSIFIER_PROMPT = """You are a classification agent. Analyze the following customer message and classify it into exactly one category:
- "positive_feedback" - if the customer is expressing satisfaction, gratitude, or praise
- "negative_feedback" - if the customer is expressing dissatisfaction, complaints, or problems
- "query" - if the customer is asking about ticket status or requesting information

Customer message: "{message}"

Respond with ONLY one word: positive_feedback, negative_feedback, or query"""

POSITIVE_FEEDBACK_PROMPT = """Generate a warm, personalized thank-you message for a customer who gave positive feedback.
Customer message: "{message}"
Customer name: {customer_name}

Format: Thank you for your kind words, [CustomerName]! We're delighted to assist you.
Keep it brief and genuine."""

class MultiAgentChat:
    def __init__(self, api_key: str, support_tickets: dict):
        """Initialize the multi-agent chat system"""
        self.llm = ChatAnthropic(api_key=api_key, model="claude-haiku-4-5-20251001")
        self.support_tickets = support_tickets
        self.graph = self._build_graph()
    
    def _get_last_user_message(self, messages):
        """Extract the last user message from the messages list"""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
            elif isinstance(msg, HumanMessage):
                return msg.content
        return ""
    
    # === AGENT 1: CLASSIFIER AGENT ===
    def classifier_agent(self, state: State):
        """Classifies user input into Positive Feedback, Negative Feedback, or Query"""
        user_message = self._get_last_user_message(state["messages"])
        
        classification_prompt = CLASSIFIER_PROMPT.format(message=user_message)
        
        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        classification = response.content.strip().lower()
        
        # Validate classification
        valid_classifications = ["positive_feedback", "negative_feedback", "query"]
        if classification not in valid_classifications:
            classification = "query"  # Default to query if unclear
        
        return {"classification": classification}
    
    # === AGENT 2: FEEDBACK HANDLER AGENT ===
    def positive_feedback_handler(self, state: State):
        """Handles positive feedback with personalized thank-you message"""
        user_message = self._get_last_user_message(state["messages"])
        customer_name = state.get("customer_name", "valued customer")
        
        feedback_prompt = POSITIVE_FEEDBACK_PROMPT.format(
            message=user_message,
            customer_name=customer_name
        )
        
        response = self.llm.invoke([HumanMessage(content=feedback_prompt)])
        
        return {"messages": [AIMessage(content=response.content)]}
    
    def negative_feedback_handler(self, state: State):
        """Handles negative feedback by generating ticket and storing in database"""
        user_message = self._get_last_user_message(state["messages"])
        
        # Generate unique 6-digit ticket number
        ticket_number = str(random.randint(100000, 999999))
        while ticket_number in self.support_tickets:
            ticket_number = str(random.randint(100000, 999999))
        
        # Store in database
        self.support_tickets[ticket_number] = {
            "status": "unresolved",
            "message": user_message,
            "type": "negative_feedback",
            "customer_name": state.get("customer_name", "Unknown")
        }
        
        response_message = f"We apologize for the inconvenience. A new ticket #{ticket_number} has been generated, and our team will follow up shortly."
        
        return {
            "messages": [AIMessage(content=response_message)],
            "ticket_number": ticket_number
        }
    
    # === AGENT 3: QUERY HANDLER AGENT ===
    def query_handler(self, state: State):
        """Handles queries by extracting ticket number and checking status"""
        user_message = self._get_last_user_message(state["messages"])
        
        # Extract ticket number using regex
        ticket_match = re.search(r'#?(\d{6})', user_message)
        
        if not ticket_match:
            response_message = "I couldn't find a ticket number in your message. Please provide a 6-digit ticket number (e.g., #123456)."
            return {"messages": [AIMessage(content=response_message)]}
        
        ticket_number = ticket_match.group(1)
        
        # Query database
        if ticket_number in self.support_tickets:
            ticket_info = self.support_tickets[ticket_number]
            ticket_status = ticket_info["status"]
            response_message = f"Your ticket #{ticket_number} is currently marked as: {ticket_status}."
        else:
            response_message = f"Ticket #{ticket_number} was not found in our system. Please verify the ticket number."
        
        return {"messages": [AIMessage(content=response_message)]}
    
    # === ROUTING LOGIC ===
    def route_after_classification(self, state: State) -> Literal["positive_feedback", "negative_feedback", "query"]:
        """Routes to appropriate handler based on classification"""
        classification = state.get("classification", "query")
        
        if classification == "positive_feedback":
            return "positive_feedback"
        elif classification == "negative_feedback":
            return "negative_feedback"
        else:
            return "query"
    
    # === BUILD GRAPH ===
    def _build_graph(self):
        """Builds the LangGraph workflow"""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("classifier", self.classifier_agent)
        graph_builder.add_node("positive_feedback", self.positive_feedback_handler)
        graph_builder.add_node("negative_feedback", self.negative_feedback_handler)
        graph_builder.add_node("query", self.query_handler)
        
        # Add edges
        graph_builder.add_edge(START, "classifier")
        graph_builder.add_conditional_edges(
            "classifier",
            self.route_after_classification,
            {
                "positive_feedback": "positive_feedback",
                "negative_feedback": "negative_feedback",
                "query": "query"
            }
        )
        graph_builder.add_edge("positive_feedback", END)
        graph_builder.add_edge("negative_feedback", END)
        graph_builder.add_edge("query", END)
        
        return graph_builder.compile()
    
    def process_message(self, messages: list, customer_name: str = "valued customer"):
        """Process a user message through the multi-agent system"""
        # Convert dict messages to LangChain messages
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            else:
                langchain_messages.append(msg)
        
        result = self.graph.invoke({
            "messages": langchain_messages,
            "customer_name": customer_name,
            "classification": "",
            "ticket_number": ""
        })
        
        return result
