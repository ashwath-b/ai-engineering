# evals/test_cases.py
TEST_CASES = [
    {
        "user_id": "user_123",
        "ip_address": "192.168.1.100",
        "expected_decision": "BLOCK",
        "expected_risk": "HIGH",
        "description": "Clear fraudster — high returns, VPN, new account"
    },
    {
        "user_id": "user_456",
        "ip_address": "203.0.113.42",
        "expected_decision": "APPROVE",
        "expected_risk": "LOW",
        "description": "Legitimate user — clean signals"
    },
    # Add 18 more cases
]