"""
Single-File LangSmith Evaluation for Multi-Agent Customer Support

This script provides a complete evaluation system using LangSmith to test
your multi-agent customer support chatbot. It uses Streamlit secrets for API keys.

Setup:
1. Add to .streamlit/secrets.toml:
   api_key = "your-anthropic-key"
   langsmith_api_key = "your-langsmith-key"

2. Run evaluation:
   python evaluation_single.py

3. View results at: https://smith.langchain.com/

Usage:
   python evaluation_single.py                    # Run basic evaluation
   python evaluation_single.py --advanced         # Run with all metrics
   python evaluation_single.py --report           # Generate local report
   python evaluation_single.py --compare ID1 ID2  # Compare two runs
"""

import os
import sys
import re
import json
import random
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Literal
from typing_extensions import TypedDict
from pathlib import Path

# Import LangSmith and LangChain
from langsmith import Client, traceable, evaluate
from langsmith.schemas import Example, Run
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

# Import the chat system
from chat import MultiAgentChat

# Try to import streamlit for secrets, fall back to env vars
try:
    import streamlit as st
    USE_STREAMLIT_SECRETS = True
except ImportError:
    USE_STREAMLIT_SECRETS = False
    print("⚠️  Streamlit not available, using environment variables")


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_NAME = "customer-support-evaluation"
PROJECT_NAME = "customer-support-evaluation"
# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_keys():
    """Get API keys from Streamlit secrets or environment variables"""
    anthropic_api_key = None
    langsmith_key = None
    
    if USE_STREAMLIT_SECRETS:
        try:
            # Try to load from secrets.toml
            anthropic_api_key = st.secrets.get("anthropic_api_key")
            langsmith_key = st.secrets.get("langsmith_api_key")
            
            if anthropic_api_key:
                print("✅ Loaded Anthropic API key from secrets.toml")
            if langsmith_key:
                print("✅ Loaded LangSmith API key from secrets.toml")
        except Exception as e:
            print(f"⚠️  Could not load from secrets.toml: {e}")
    
    # Fall back to environment variables
    if not anthropic_api_key:
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            print("✅ Loaded Anthropic API key from environment")
    
    if not langsmith_key:
        langsmith_key = os.environ.get("LANGCHAIN_API_KEY")
        if langsmith_key:
            print("✅ Loaded LangSmith API key from environment")
    
    return anthropic_api_key, langsmith_key


def setup_langsmith():
    """Setup LangSmith environment"""
    _, langsmith_key = get_api_keys()
    
    if not langsmith_key:
        print("\n❌ Error: LangSmith API key not found!")
        print("   Add to .streamlit/secrets.toml:")
        print('   langsmith_api_key = "your-key"')
        print("\n   Or set environment variable:")
        print('   export LANGCHAIN_API_KEY="your-key"')
        print("\n   Get your key from: https://smith.langchain.com/")
        sys.exit(1)
    
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME


# ============================================================================
# TEST DATASET
# ============================================================================

def create_test_dataset():
    """Create comprehensive test dataset"""
    
    test_cases = [
        # === POSITIVE FEEDBACK TESTS ===
        {
            "inputs": {
                "message": "Thank you so much for your excellent service!",
                "customer_name": "John Doe"
            },
            "outputs": {
                "expected_classification": "positive_feedback",
                "should_contain": ["thank", "John Doe"],
                "should_not_create_ticket": True
            }
        },
        {
            "inputs": {
                "message": "I'm really happy with how you handled my issue. Great work!",
                "customer_name": "Jane Smith"
            },
            "outputs": {
                "expected_classification": "positive_feedback",
                "should_contain": ["Jane Smith"],
                "should_not_create_ticket": True
            }
        },
        {
            "inputs": {
                "message": "Amazing support team! You guys are the best!",
                "customer_name": "Bob Johnson"
            },
            "outputs": {
                "expected_classification": "positive_feedback",
                "should_contain": ["Bob Johnson"],
                "should_not_create_ticket": True
            }
        },
        
        # === NEGATIVE FEEDBACK TESTS ===
        {
            "inputs": {
                "message": "I'm very disappointed with the service. Nothing is working.",
                "customer_name": "Alice Brown"
            },
            "outputs": {
                "expected_classification": "negative_feedback",
                "should_contain": ["ticket", "apologize"],
                "should_create_ticket": True,
                "ticket_should_be_unresolved": True
            }
        },
        {
            "inputs": {
                "message": "This is unacceptable! My order hasn't arrived and nobody is helping me.",
                "customer_name": "Charlie Wilson"
            },
            "outputs": {
                "expected_classification": "negative_feedback",
                "should_contain": ["ticket"],
                "should_create_ticket": True,
                "ticket_should_be_unresolved": True
            }
        },
        {
            "inputs": {
                "message": "I've been waiting for 3 days and still no response. Very frustrated!",
                "customer_name": "Diana Martinez"
            },
            "outputs": {
                "expected_classification": "negative_feedback",
                "should_contain": ["ticket"],
                "should_create_ticket": True
            }
        },
        
        # === QUERY TESTS ===
        {
            "inputs": {
                "message": "What's the status of ticket #123456?",
                "customer_name": "Eve Davis",
                "setup_ticket": "123456"
            },
            "outputs": {
                "expected_classification": "query",
                "should_contain": ["123456", "status"],
                "should_not_create_ticket": True
            }
        },
        {
            "inputs": {
                "message": "Can you check ticket 789012 for me?",
                "customer_name": "Frank Miller",
                "setup_ticket": "789012"
            },
            "outputs": {
                "expected_classification": "query",
                "should_contain": ["789012"],
                "should_not_create_ticket": True
            }
        },
        {
            "inputs": {
                "message": "I want to know about my ticket 555555",
                "customer_name": "Grace Lee",
                "setup_ticket": "555555"
            },
            "outputs": {
                "expected_classification": "query",
                "should_contain": ["555555"],
                "should_not_create_ticket": True
            }
        },
        
        # === EDGE CASES ===
        {
            "inputs": {
                "message": "Hello, how are you?",
                "customer_name": "Henry Clark"
            },
            "outputs": {
                "expected_classification": "query",
                "should_not_create_ticket": True
            }
        },
        {
            "inputs": {
                "message": "Check ticket 999999",
                "customer_name": "Ivy White",
                "setup_ticket": None
            },
            "outputs": {
                "expected_classification": "query",
                "should_contain": ["999999", "not found"],
                "should_not_create_ticket": True
            }
        },
        {
            "inputs": {
                "message": "What's the status of my ticket?",
                "customer_name": "Jack Harris"
            },
            "outputs": {
                "expected_classification": "query",
                "should_contain": ["ticket number"],
                "should_not_create_ticket": True
            }
        }
    ]
    
    return test_cases


# ============================================================================
# AGENT EXECUTION
# ============================================================================

@traceable(name="customer_support_agent")
def run_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run the multi-agent system with given inputs (traced by LangSmith)"""
    
    anthropic_api_key, _ = get_api_keys()
    
    if not anthropic_api_key:
        raise ValueError("Anthropic API key not found in secrets or environment")
    
    # Initialize support tickets database
    support_tickets = {}
    
    # Setup ticket if specified
    if "setup_ticket" in inputs and inputs["setup_ticket"]:
        ticket_num = inputs["setup_ticket"]
        support_tickets[ticket_num] = {
            "status": "unresolved",
            "message": "Test ticket",
            "type": "test",
            "customer_name": inputs.get("customer_name", "Test User")
        }
    
    # Initialize chat system
    chat_system = MultiAgentChat(anthropic_api_key, support_tickets)
    
    # Process message
    message = inputs["message"]
    customer_name = inputs.get("customer_name", "Customer")
    
    messages = [{"role": "user", "content": message}]
    result = chat_system.process_message(messages, customer_name)
    
    # Extract response
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        response_text = last_message.content
    elif isinstance(last_message, dict):
        response_text = last_message["content"]
    else:
        response_text = str(last_message)
    
    return {
        "response": response_text,
        "classification": result.get("classification", ""),
        "ticket_number": result.get("ticket_number", ""),
        "support_tickets": dict(support_tickets)
    }


# ============================================================================
# EVALUATORS
# ============================================================================

def classification_accuracy_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate classification accuracy"""
    expected = example.outputs.get("expected_classification")
    if not expected:
        return {"key": "classification_accuracy", "score": None}
    
    actual = run.outputs.get("classification", "").lower()
    score = 1.0 if actual == expected else 0.0
    
    return {
        "key": "classification_accuracy",
        "score": score,
        "comment": f"Expected: {expected}, Got: {actual}"
    }


def response_quality_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate response quality"""
    response = run.outputs.get("response", "").lower()
    expected_outputs = example.outputs
    
    score = 1.0
    issues = []
    
    # Check required content
    should_contain = expected_outputs.get("should_contain", [])
    for phrase in should_contain:
        if phrase.lower() not in response:
            score -= 0.3
            issues.append(f"Missing: '{phrase}'")
    
    # Check forbidden content
    should_not_contain = expected_outputs.get("should_not_contain", [])
    for phrase in should_not_contain:
        if phrase.lower() in response:
            score -= 0.3
            issues.append(f"Contains: '{phrase}'")
    
    score = max(0.0, min(1.0, score))
    comment = " | ".join(issues) if issues else "Quality good"
    
    return {
        "key": "response_quality",
        "score": score,
        "comment": comment
    }


def ticket_management_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate ticket management"""
    expected_outputs = example.outputs
    support_tickets = run.outputs.get("support_tickets", {})
    ticket_number = run.outputs.get("ticket_number", "")
    
    score = 1.0
    issues = []
    
    should_create = expected_outputs.get("should_create_ticket", False)
    should_not_create = expected_outputs.get("should_not_create_ticket", False)
    
    ticket_created = bool(ticket_number)
    
    if should_create and not ticket_created:
        score = 0.0
        issues.append("Expected ticket creation")
    elif should_not_create and ticket_created:
        score = 0.0
        issues.append("Unexpected ticket creation")
    
    # Check ticket properties
    if ticket_created and ticket_number in support_tickets:
        ticket = support_tickets[ticket_number]
        
        if expected_outputs.get("ticket_should_be_unresolved", False):
            if ticket["status"] != "unresolved":
                score -= 0.3
                issues.append(f"Status: {ticket['status']}")
        
        customer_name = example.inputs.get("customer_name", "")
        if customer_name and ticket.get("customer_name") != customer_name:
            score -= 0.2
            issues.append("Customer name mismatch")
    
    score = max(0.0, score)
    comment = " | ".join(issues) if issues else "Ticket management correct"
    
    return {
        "key": "ticket_management",
        "score": score,
        "comment": comment
    }


def response_length_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate response length"""
    response = run.outputs.get("response", "")
    length = len(response)
    
    if length < 20:
        score = 0.3
        comment = f"Too short ({length} chars)"
    elif length > 400:
        score = 0.7
        comment = f"Too long ({length} chars)"
    else:
        score = 1.0
        comment = f"Appropriate ({length} chars)"
    
    return {
        "key": "response_length",
        "score": score,
        "comment": comment
    }


# ============================================================================
# ADVANCED EVALUATORS
# ============================================================================

def empathy_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate empathy in response"""
    response = run.outputs.get("response", "").lower()
    classification = run.outputs.get("classification", "")
    
    positive_indicators = [
        "thank you", "we appreciate", "we're delighted", "sorry",
        "apologize", "we understand", "inconvenience", "we'll help"
    ]
    
    negative_indicators = [
        "you should have", "you need to", "you must", "your fault"
    ]
    
    score = 0.5
    empathy_found = sum(1 for ind in positive_indicators if ind in response)
    negative_found = sum(1 for ind in negative_indicators if ind in response)
    
    if empathy_found > 0:
        score += min(0.3, empathy_found * 0.15)
    if negative_found > 0:
        score -= min(0.5, negative_found * 0.25)
    
    if classification == "negative_feedback" and any(w in response for w in ["sorry", "apologize"]):
        score += 0.2
    
    score = max(0.0, min(1.0, score))
    
    return {
        "key": "empathy_score",
        "score": score,
        "comment": f"Positive: {empathy_found}, Negative: {negative_found}"
    }


def personalization_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate personalization"""
    response = run.outputs.get("response", "")
    customer_name = example.inputs.get("customer_name", "")
    
    score = 0.5
    issues = []
    
    if customer_name and customer_name in response:
        score += 0.3
    elif customer_name:
        issues.append("Name not used")
        score -= 0.2
    
    generic_phrases = ["dear customer", "hello user", "hi there"]
    if any(phrase in response.lower() for phrase in generic_phrases):
        score -= 0.1
        issues.append("Generic greeting")
    
    score = max(0.0, min(1.0, score))
    comment = " | ".join(issues) if issues else "Good personalization"
    
    return {
        "key": "personalization_score",
        "score": score,
        "comment": comment
    }


def consistency_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate consistency between classification and response"""
    classification = run.outputs.get("classification", "")
    response = run.outputs.get("response", "").lower()
    
    score = 1.0
    issues = []
    
    if classification == "positive_feedback":
        if not any(word in response for word in ["thank", "appreciate", "delighted"]):
            score -= 0.4
            issues.append("Missing gratitude")
    
    elif classification == "negative_feedback":
        if "ticket" not in response:
            score -= 0.3
            issues.append("No ticket mentioned")
        if not any(word in response for word in ["sorry", "apologize"]):
            score -= 0.2
            issues.append("No apology")
    
    elif classification == "query":
        if "ticket" not in response and "status" not in response:
            score -= 0.3
            issues.append("Doesn't address query")
    
    score = max(0.0, score)
    comment = " | ".join(issues) if issues else "Consistent"
    
    return {
        "key": "consistency_score",
        "score": score,
        "comment": comment
    }


def llm_judge_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Use Claude to evaluate response quality"""
    anthropic_api_key, _ = get_api_keys()
    
    if not anthropic_api_key:
        return {"key": "llm_judge_quality", "score": None, "comment": "No API key"}
    
    llm = ChatAnthropic(api_key=anthropic_api_key, model="claude-sonnet-4-5-20250929")
    
    customer_message = example.inputs.get("message", "")
    agent_response = run.outputs.get("response", "")
    expected_classification = example.outputs.get("expected_classification", "")
    
    evaluation_prompt = f"""Evaluate this customer support response on a scale of 0-10.

Customer: "{customer_message}"
Expected Type: {expected_classification}
Agent: "{agent_response}"

Rate on: appropriateness, helpfulness, professionalism, completeness.

Respond ONLY with JSON:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""

    try:
        response = llm.invoke([HumanMessage(content=evaluation_prompt)])
        response_text = response.content.strip()
        
        # Clean JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        score = result["score"] / 10.0
        reasoning = result["reasoning"]
        
        return {
            "key": "llm_judge_quality",
            "score": score,
            "comment": reasoning
        }
    except Exception as e:
        return {
            "key": "llm_judge_quality",
            "score": None,
            "comment": f"Failed: {str(e)}"
        }


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

def create_langsmith_dataset(client: Client):
    """Create or update LangSmith dataset"""
    
    # Delete existing dataset
    try:
        client.delete_dataset(dataset_name=DATASET_NAME)
        print(f"Deleted existing dataset: {DATASET_NAME}")
    except:
        pass
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Test cases for multi-agent customer support"
    )
    
    # Add test cases
    test_cases = create_test_dataset()
    for test_case in test_cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs=test_case["inputs"],
            outputs=test_case["outputs"]
        )
    
    print(f"✅ Created dataset with {len(test_cases)} test cases")
    return dataset


# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation(advanced: bool = False, run_name: str = None):
    """Run the evaluation"""
    
    setup_langsmith()
    client = Client()
    
    print("\n" + "="*70)
    print("🚀 Starting LangSmith Evaluation")
    print("="*70 + "\n")
    
    # Create dataset
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"📊 Using existing dataset: {DATASET_NAME}")
    except:
        print(f"📊 Creating dataset: {DATASET_NAME}")
        dataset = create_langsmith_dataset(client)
    
    # Select evaluators
    evaluators = [
        classification_accuracy_evaluator,
        response_quality_evaluator,
        ticket_management_evaluator,
        response_length_evaluator
    ]
    
    if advanced:
        print("🎯 Advanced mode: Using all 7 metrics")
        evaluators.extend([
            empathy_evaluator,
            personalization_evaluator,
            consistency_evaluator,
            llm_judge_evaluator
        ])
    else:
        print("⚡ Basic mode: Using 4 core metrics")
    
    print(f"\n🧪 Running evaluation...\n")
    
    # Run evaluation
    results = evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=run_name or ("advanced-eval" if advanced else "eval"),
        description="Multi-agent customer support evaluation",
        max_concurrency=1 if advanced else 2
    )
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)
    print(f"\n📈 View results: https://smith.langchain.com/")
    print(f"   Project: {PROJECT_NAME}\n")
    
    return results


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(days_back: int = 7):
    """Generate evaluation report"""
    
    setup_langsmith()
    client = Client()
    
    print("\n" + "="*70)
    print("📈 Evaluation Report")
    print("="*70)
    
    start_time = datetime.now() - timedelta(days=days_back)
    
    try:
        runs = list(client.list_runs(
            project_name=PROJECT_NAME,
            start_time=start_time,
            execution_order=1
        ))
        
        print(f"\n📅 Last {days_back} days")
        print(f"🔢 Total runs: {len(runs)}\n")
        print("-"*70)
        
        for i, run in enumerate(runs[:10], 1):  # Show first 10
            print(f"\n{i}. {run.name}")
            print(f"   Time: {run.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   ID: {run.id}")
            
            # Get feedback
            feedbacks = list(client.list_feedback(run_ids=[str(run.id)]))
            if feedbacks:
                print("   Metrics:")
                metrics = {}
                for fb in feedbacks:
                    if fb.score is not None:
                        if fb.key not in metrics:
                            metrics[fb.key] = []
                        metrics[fb.key].append(fb.score)
                
                for key, scores in metrics.items():
                    avg = sum(scores) / len(scores)
                    print(f"   • {key}: {avg:.2f}")
        
        print("\n" + "-"*70)
        print(f"\n📈 Full details: https://smith.langchain.com/")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def compare_runs(run_id_1: str, run_id_2: str):
    """Compare two evaluation runs"""
    
    setup_langsmith()
    client = Client()
    
    print("\n" + "="*70)
    print("🔄 Comparing Runs")
    print("="*70 + "\n")
    
    def get_metrics(run_id):
        feedbacks = list(client.list_feedback(run_ids=[run_id]))
        metrics = {}
        for fb in feedbacks:
            if fb.score is not None:
                if fb.key not in metrics:
                    metrics[fb.key] = []
                metrics[fb.key].append(fb.score)
        return {k: sum(v)/len(v) for k, v in metrics.items()}
    
    try:
        metrics_1 = get_metrics(run_id_1)
        metrics_2 = get_metrics(run_id_2)
        
        print(f"Run 1: {run_id_1}")
        print(f"Run 2: {run_id_2}\n")
        print("-"*70)
        print(f"{'Metric':<30} {'Run 1':>12} {'Run 2':>12} {'Diff':>12}")
        print("-"*70)
        
        all_keys = set(metrics_1.keys()) | set(metrics_2.keys())
        for key in sorted(all_keys):
            val1 = metrics_1.get(key, 0)
            val2 = metrics_2.get(key, 0)
            diff = val2 - val1
            
            diff_str = f"{diff:+.3f}"
            if diff > 0.05:
                diff_str += " 📈"
            elif diff < -0.05:
                diff_str += " 📉"
            
            print(f"{key:<30} {val1:>12.3f} {val2:>12.3f} {diff_str:>12}")
        
        print("-"*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LangSmith Evaluation for Multi-Agent Customer Support"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced evaluation with all metrics"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this evaluation run"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate evaluation report"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to include in report (default: 7)"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_ID_1", "RUN_ID_2"),
        help="Compare two evaluation runs"
    )
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create/recreate the evaluation dataset"
    )
    
    args = parser.parse_args()
    
    # Check API keys
    anthropic_api_key, langsmith_key = get_api_keys()
    
    if not anthropic_api_key:
        print("\n❌ Error: Anthropic API key not found!")
        print("   Add to .streamlit/secrets.toml:")
        print('   api_key = "your-anthropic-key"')
        print("\n   Or set environment variable:")
        print('   export ANTHROPIC_API_KEY="your-key"')
        sys.exit(1)
    
    if not langsmith_key:
        print("\n❌ Error: LangSmith API key not found!")
        print("   Add to .streamlit/secrets.toml:")
        print('   langsmith_api_key = "your-langsmith-key"')
        print("\n   Or set environment variable:")
        print('   export LANGCHAIN_API_KEY="your-key"')
        print("\n   Get your key from: https://smith.langchain.com/")
        sys.exit(1)
    
    # Execute requested action
    if args.create_dataset:
        setup_langsmith()
        create_langsmith_dataset(Client())
    elif args.compare:
        compare_runs(args.compare[0], args.compare[1])
    elif args.report:
        generate_report(args.days)
    else:
        run_evaluation(args.advanced, args.run_name)


if __name__ == "__main__":
    main()
