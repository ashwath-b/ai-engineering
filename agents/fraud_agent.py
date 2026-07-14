# agents/fraud_agent.py
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError
from rag.query import retrieve
from rag.langchain_rag import ask
from core.config import GROQ_API_KEY
import os
import json

DISPOSABLE_DOMAINS = ["tempmail.com", "disposablemail.com"]
KNOWN_FRAUD_EMAILS = ["john@doe.com", "abc@xyz.com", "john@tempmail.com", "scammer@gmail.com"]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,       # low — fraud decisions need consistency
    api_key = GROQ_API_KEY
)

# ── Tools — things the agent can DO ───────────────────────────────────────────
# For now these are mock tools — fake data
# Next week we'll make them call real APIs

@tool
def get_transaction_history(user_id: str) -> str:
    """Get recent transaction history for a user"""
    # Mock data — replace with real DB query later
    mock_data = {
        "user_123": {
            "total_transactions": 45,
            "total_returns": 38,
            "return_rate": "84%",
            "avg_dollars_per_return": 245,
            "non_receipted_return_rate": "71%",
            "unique_stores_visited": 12,
            "recent_transactions": [
                {"date": "2024-01-15", "amount": 450, "type": "purchase"},
                {"date": "2024-01-16", "amount": 445, "type": "return", "receipted": False},
                {"date": "2024-01-17", "amount": 380, "type": "purchase"},
                {"date": "2024-01-17", "amount": 375, "type": "return", "receipted": False},
            ]
        },
        "user_456": {
            "total_transactions": 12,
            "total_returns": 2,
            "return_rate": "16%",
            "avg_dollars_per_return": 45,
            "non_receipted_return_rate": "0%",
            "unique_stores_visited": 2,
            "recent_transactions": [
                {"date": "2024-01-10", "amount": 89, "type": "purchase"},
                {"date": "2024-01-20", "amount": 45, "type": "return", "receipted": True},
            ]
        }
    }
    data = mock_data.get(user_id, {"error": "User not found"})
    return json.dumps(data)

@tool
def check_ip_reputation(ip_address: str) -> str:
    """Check if an IP address is associated with fraud"""
    # Mock data
    risky_ips = ["192.168.1.100", "10.0.0.55"]
    is_risky = ip_address in risky_ips
    return json.dumps({
        "ip": ip_address,
        "risk_level": "HIGH" if is_risky else "LOW",
        "known_fraud": is_risky,
        "country": "US",
        "vpn_detected": is_risky
    })

@tool
def get_account_info(user_id: str) -> str:
    """Get account creation date and basic info"""
    mock_accounts = {
        "user_123": {
            "account_age_days": 3,
            "email_verified": False,
            "email": "john@tempmail.com",
            "phone_verified": False,
            "account_flags": ["multiple_failed_logins", "password_reset_requested"]
        },
        "user_456": {
            "account_age_days": 365,
            "email_verified": True,
            "email": "aab@vada.com",
            "phone_verified": True,
            "account_flags": []
        }
    }
    data = mock_accounts.get(user_id, {"error": "Account not found"})
    return json.dumps(data)

@tool
def calculate_fraud_score(
    return_rate: float,
    account_age_days: int,
    non_receipted_rate: float,
    unique_stores: int
) -> str:
    """Calculate a fraud risk score based on key indicators"""
    score = 0

    if return_rate > 0.7:   score += 40
    elif return_rate > 0.4: score += 20

    if account_age_days < 7:    score += 30
    elif account_age_days < 30: score += 15

    if non_receipted_rate > 0.5: score += 20
    elif non_receipted_rate > 0.2: score += 10

    if unique_stores > 10:  score += 10
    elif unique_stores > 5: score += 5

    risk_level = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"

    return json.dumps({
        "fraud_score": score,
        "risk_level": risk_level,
        "max_score": 100
    })

@tool
def search_fraud_policy(query: str) -> str:
    """Search internal fraud policy documents for relevant guidelines
    and thresholds. Returns raw policy excerpts. Use this for quick
    lookups of specific fraud indicators, thresholds, or patterns
    mentioned in policy documents. Call this during every investigation
    to check if the user's behavior matches known fraud patterns."""
    chunks = retrieve(query, n_results=3)
    if not chunks:
        return "No relevant policy found for this query."
    return "\n\n---\n\n".join(c.text for c in chunks)

@tool
def ask_fraud_policy(question: str) -> str:
    """Ask a natural language question about fraud policy and get a
    synthesized answer. Use this when you need a clear explanation
    of what policy says about a specific situation — not just raw text.
    More thorough than search_fraud_policy but slower."""
    answer = ask(question)
    if not answer:
        return "Could not find an answer in policy documents."
    return answer

@tool
def check_email_reputation(email: str) -> str:
    """Check if an emailaddress is associated with fraud by Searching
    internal list if emails tagged as known fraudulent addresses or disposable domains.
    It accepts an email address and returns a JSON object with disposable_domain,
    known_fraud & risk_level. WHen the emailaddress is malformed it returns an error.
    Call this when an emailaddress isn't verified and is suspicious."""
    # Mock data
    if "@" not in email or "." not in email.split("@")[-1]:
        return json.dumps({
            "error": "Malformed email address"
        })
    domain = email.split("@")[-1]
    disposable_domain = domain in DISPOSABLE_DOMAINS
    known_fraud = email in KNOWN_FRAUD_EMAILS
    risk_level = "HIGH" if known_fraud or disposable_domain else "LOW"
    return json.dumps({
        "email": email,
        "risk_level": risk_level,
        "known_fraud": known_fraud,
        "disposable_domain": disposable_domain
    })

@tool
def get_linked_accounts(user_id: str) -> str:
    """Find accounts linked to other user/accounts by shared device or IP
    It accepts user_id and returns JSON with linked_accounts (array)
    Each linked_account has user_id, link_types (array), account_age_days, last_seen.
    If no linked accounts found, returns an empty array.
    Call this when a user is suspected of fraud to see if they are connected
    to other accounts that have been flagged for suspicious activity."""
    # Mock data
    mock_linked_accounts = {
        "user_123": {
            "linked_accounts": [
                {
                    "user_id": "user_789",
                    "link_types": ["shared_device"],
                    "account_age_days": 2,
                    "last_seen": "2024-01-18"
                },
                {
                    "user_id": "user_555",
                    "link_types": ["shared_ip", "shared_device"],
                    "account_age_days": 5,
                    "last_seen": "2024-01-17"
                }
            ]
        },

        "user_456": {
            "linked_accounts": []
        }
    }
    user = mock_linked_accounts.get(user_id)

    if user is None:
        return json.dumps({"error": "User not found"})

    linked_accounts = user.get("linked_accounts", [])
    return json.dumps({
        "linked_accounts": linked_accounts
    })

@tool
def check_transaction_velocity(user_id: str) -> str:
    """Find if a user account has unusually high number of transactions
    in a short period of time. Accepts user_id and returns a JSON object with
    transactions_last_hour, returns_last_hour, highest_transactions_in_any_hour,
    days_since_last_transaction and velocity_risk (high/low).
    Call this when investigating users with high return rates
    or suspicious activity."""
    # Mock data
    mock_velocity = {

        "user_123": {
            "transactions_last_hour": 14,
            "returns_last_hour": 9,
            "highest_transactions_in_any_hour": 17,
            "days_since_last_transaction": 0,
            "velocity_risk": "HIGH"
        },

        "user_456": {
            "transactions_last_hour": 1,
            "returns_last_hour": 0,
            "highest_transactions_in_any_hour": 2,
            "days_since_last_transaction": 6,
            "velocity_risk": "LOW"
        }
    }

    data = mock_velocity.get(user_id, {"error": "User not found"})
    return json.dumps(data)
# ── Prompt ──────────────────────────────────────────────────────────────────────

system_prompt = """You are a senior fraud investigator.
Use the available tools to investigate fraud cases thoroughly.
Always use all relevant tools before writing your final report.

INVESTIGATION ORDER:
1. Call get_transaction_history, check_ip_reputation, get_account_info,
   get_linked_accounts, check_transaction_velocity in parallel — gather all user data first
2. After get_account_info returns, call check_email_reputation with the
   exact "email" value from its output. Never guess or construct an
   email address. If account info contains no email, skip this check
   and note that in the report.
3. Call search_fraud_policy with relevant queries to check if user
   behavior matches known fraud patterns in our policy documents
4. Call calculate_fraud_score using ACTUAL values from step 1
5. Synthesize all findings into a structured report

IMPORTANT - Follow this exact order:
- Always search fraud policy documents during every investigation
- Never call calculate_fraud_score before receiving data from step 1
- Use actual retrieved values, never assumed or default values

Never call calculate_fraud_score with assumed or default values.
"""

# ── Agent ──────────────────────────────────────────────────────────────────────

tools = [
    get_transaction_history,
    check_ip_reputation,
    get_account_info,
    calculate_fraud_score,
    search_fraud_policy,
    ask_fraud_policy,
    check_email_reputation,
    get_linked_accounts,
    check_transaction_velocity
]

agent = create_agent(llm, tools, system_prompt=system_prompt)

# ── Run investigation ──────────────────────────────────────────────────────────

def investigate(user_id: str, ip_address: str) -> str:
    """Run a full fraud investigation on a user"""

    prompt = f"""
You are a senior fraud investigator. Investigate the following case thoroughly.

Case Details:
- User ID: {user_id}
- Login IP: {ip_address}

Instructions:
1. Get transaction history for the user, investigate linked accounts & check_transaction_velocity
2. Check IP reputation & emailaddress is fraudulent or disposable
3. Get account information
4. Calculate fraud score using the retrieved data
5. Synthesize all findings into a structured report

Your final report MUST include:
- User ID: (the user being investigated)
- Login IP: (the IP address checked)
- Email Address: (If available, else "N/A")
- Risk Level: HIGH / MEDIUM / LOW
- Key Fraud Signals Found: (list each signal)
- Policy References: (what policy says about these signals)
- Recommended Action: BLOCK / REVIEW / APPROVE
- Confidence: 0-100%
- Brief Justification
"""
    try:
      result = agent.invoke(
          {
              'messages': [HumanMessage(content=prompt)]
          },
          config={'recursion_limit': 10}
      )

      # Get the last message — agent's final response
      return result["messages"][-1].content
    except GraphRecursionError as e:
        print(f'Crashed as predicted: GraphRecursionError')
    except Exception as e:
        print(f'Different error: {type(e).__name__}: {e}')

# ── Test ───────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("=== Investigating HIGH RISK user ===")
#     print(investigate("user_123", "192.168.1.100"))

#     print("\n\n=== Investigating LOW RISK user ===")
#     print(investigate("user_456", "203.0.113.42"))