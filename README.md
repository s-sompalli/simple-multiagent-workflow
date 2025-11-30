# Simple Multi-Agent Evaluation Example (using LangSmith)

The example uses a ticket system to track issues via multiple agents

## Guide

### Step 1: Setup Secrets File

Create `.streamlit/secrets.toml` in your project (sample empty file is here)

```secrets.toml
# Get from https://console.anthropic.com/
api_key = "sk-ant-your-key-here"

# Get from https://smith.langchain.com/
langsmith_api_key = "lsv2_pt_your-key-here"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run app.py (to use the main workflow) or evaluate.py

```bash
streamlit run app.py
python evaluate.py
```

View results at https://smith.langchain.com/ using your account

## Evaluation Script Does The Following:

- Tests your agent with 12 scenarios
- Measures 4 core metrics (8 with --advanced flag)
- Uploads results to LangSmith
- Gives you detailed performance data (requires plus plan)

## 🎯 Common Commands

```bash
# Basic evaluation
python evaluate.py

# All 7 metrics
python evaluate.py --advanced

# Named run
python evaluate.py --run-name "baseline"

# Generate report
python evaluate.py --report

# Compare runs
python evaluate.py --compare ID1 ID2
```

## Metrics

**Basic**
- Classification Accuracy
- Response Quality
- Ticket Management
- Response Length

**Advanced**
- Empathy Score
- Personalization
- Consistency
- LLM Judge Quality

## Test Cases
**Positive Feedback**
- "Thank you for your help!"
- "Great service!"
- "You guys are amazing!"

**Negative Feedback**
- "I'm disappointed..."
- "This is unacceptable!"
- "Very frustrated!"

**Queries**
- "Status of ticket #123456?"
- "Check my ticket 789012"
- "What about ticket 555555?"

**Edge Cases**
- Unclear messages
- Invalid ticket numbers
- Missing information

