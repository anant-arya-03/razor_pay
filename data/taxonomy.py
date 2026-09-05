"""
Root-cause taxonomy for payment failures.

Grounded in Razorpay's real error schema. Every failed payment returns:
    error_code         BAD_REQUEST_ERROR | GATEWAY_ERROR | SERVER_ERROR
    error_source       customer | business | bank | gateway | internal
    error_step         payment_initiation | payment_authentication
                       | payment_authorization | payment_response
    error_reason       machine-readable reason string
    error_description  FREE TEXT, wording varies by bank/acquirer

IMPORTANT (do this before you submit):
Download the official reason list from Razorpay docs
    https://razorpay.com/docs/build/browser/assets/images/payments_error_reasons.xlsx
and replace the REASON strings below with the exact ones from that sheet.
Your DECISIONS.md should say the taxonomy is derived from Razorpay's
published error reasons -- that is a real, checkable claim.

The 6 CAUSE buckets below are OUR grouping. Razorpay publishes dozens of
reasons; we collapse them into the 6 that imply *different recovery actions*.
That collapse is the modelling decision, and it is what makes this a
classification problem rather than a dictionary lookup.
"""

CAUSES = [
    "insufficient_funds",   # money nahi tha -> retry at the right TIME
    "bank_downtime",        # bank/gateway down -> silent retry shortly
    "auth_drop_otp",        # customer left at OTP/3DS -> nudge back
    "hard_decline",         # card blocked/expired -> needs NEW instrument
    "checkout_abandoned",   # never reached payment -> nudge / offer
    "mandate_failure",      # subscription mandate failed -> re-authorize
]

CAUSE_ID = {c: i for i, c in enumerate(CAUSES)}

# ---------------------------------------------------------------------------
# Realistic field combinations per cause.
# NOTE the deliberate overlap on error_code: BAD_REQUEST_ERROR and
# GATEWAY_ERROR each appear under several causes. That overlap is the whole
# point -- it is why code-only lookup cannot solve this.
# ---------------------------------------------------------------------------
CAUSE_PROFILE = {
    "insufficient_funds": {
        "codes": [("BAD_REQUEST_ERROR", 0.85), ("GATEWAY_ERROR", 0.15)],
        "sources": [("bank", 0.6), ("customer", 0.4)],
        "steps": [("payment_authorization", 0.9), ("payment_response", 0.1)],
        "reasons": ["payment_failed", "insufficient_funds_or_limit_exceeded"],
        "methods": [("upi", 0.45), ("card", 0.35), ("netbanking", 0.20)],
        "descriptions": [
            "Your payment could not be completed due to insufficient funds in your account.",
            "Payment failed. The account does not have sufficient balance.",
            "Transaction declined by bank - insufficient balance available.",
            "Insufficient funds. Please retry with another account.",
            "The bank declined the payment as the available limit was exceeded.",
            "Payment was declined because the daily transaction limit is exhausted.",
            "Txn declined: low balance in customer account.",
            "Debit failed - balance below transaction amount.",
        ],
    },
    "bank_downtime": {
        "codes": [("GATEWAY_ERROR", 0.75), ("SERVER_ERROR", 0.25)],
        "sources": [("bank", 0.5), ("gateway", 0.35), ("internal", 0.15)],
        "steps": [("payment_authorization", 0.5), ("payment_response", 0.5)],
        "reasons": ["payment_failed", "gateway_technical_error", "server_error"],
        "methods": [("upi", 0.5), ("netbanking", 0.3), ("card", 0.2)],
        "descriptions": [
            "Payment processing failed because of a temporary issue at the bank.",
            "The bank is currently unavailable. Please try again after some time.",
            "Gateway timeout while processing the transaction.",
            "Technical error at the issuing bank. No response received.",
            "Payment failed due to a temporary technical error. Please retry.",
            "Unable to reach the bank server. Transaction could not be completed.",
            "Upstream system did not respond within the expected time.",
            "Service temporarily unavailable at the acquirer end.",
        ],
    },
    "auth_drop_otp": {
        "codes": [("BAD_REQUEST_ERROR", 0.9), ("GATEWAY_ERROR", 0.1)],
        "sources": [("customer", 0.85), ("bank", 0.15)],
        "steps": [("payment_authentication", 1.0)],
        "reasons": ["payment_failed", "invalid_otp", "authentication_failed"],
        "methods": [("card", 0.6), ("upi", 0.25), ("netbanking", 0.15)],
        "descriptions": [
            "Payment failed as the OTP entered was incorrect.",
            "Authentication failed. The customer did not complete OTP verification.",
            "The OTP was not entered within the allowed time and the session expired.",
            "3D Secure authentication was not completed by the customer.",
            "Customer cancelled the payment at the authentication step.",
            "Verification failed - wrong one time password submitted.",
            "UPI collect request expired without customer approval.",
            "Payment abandoned on the bank authentication page.",
        ],
    },
    "hard_decline": {
        "codes": [("BAD_REQUEST_ERROR", 0.95), ("GATEWAY_ERROR", 0.05)],
        "sources": [("bank", 0.6), ("customer", 0.4)],
        "steps": [("payment_authorization", 0.7), ("payment_initiation", 0.3)],
        "reasons": ["payment_failed", "card_blocked", "invalid_card_details"],
        "methods": [("card", 0.85), ("upi", 0.10), ("netbanking", 0.05)],
        "descriptions": [
            "The card has expired. Please use a different card.",
            "Card is blocked by the issuing bank.",
            "Transaction declined - the card is not permitted for online payments.",
            "Invalid card number entered.",
            "The issuing bank has permanently declined this transaction.",
            "This card does not support international or e-commerce transactions.",
            "Card reported lost or stolen. Payment refused.",
            "Payment refused by issuer - do not honour.",
        ],
    },
    "checkout_abandoned": {
        "codes": [("BAD_REQUEST_ERROR", 1.0)],
        "sources": [("customer", 1.0)],
        "steps": [("payment_initiation", 1.0)],
        "reasons": ["payment_cancelled", "checkout_closed"],
        "methods": [("upi", 0.4), ("card", 0.35), ("netbanking", 0.25)],
        "descriptions": [
            "Customer closed the checkout window before completing the payment.",
            "Payment was cancelled by the customer.",
            "The checkout session expired without a payment attempt.",
            "Order created but no payment method was selected.",
            "User exited the payment page.",
            "Session abandoned at the method selection screen.",
            "No payment attempt was made against this order.",
            "Checkout dropped before entering payment details.",
        ],
    },
    "mandate_failure": {
        "codes": [("BAD_REQUEST_ERROR", 0.7), ("GATEWAY_ERROR", 0.3)],
        "sources": [("bank", 0.55), ("customer", 0.3), ("gateway", 0.15)],
        "steps": [("payment_authorization", 0.6), ("payment_initiation", 0.4)],
        "reasons": ["payment_failed", "mandate_revoked", "mandate_not_found"],
        "methods": [("emandate", 0.5), ("upi", 0.3), ("card", 0.2)],
        "descriptions": [
            "The mandate has been revoked by the customer.",
            "Recurring debit failed because the mandate is no longer active.",
            "The registered e-mandate could not be found for this subscription.",
            "Auto debit failed - standing instruction not honoured by the bank.",
            "Subscription charge failed. The mandate has expired.",
            "UPI autopay mandate is paused and cannot be debited.",
            "Recurring payment rejected - customer has cancelled the standing instruction.",
            "Mandate debit declined by the sponsor bank.",
        ],
    },
}

# ---------------------------------------------------------------------------
# What the code-only lookup baseline would guess (majority class per code).
# Deliberately dumb -- this is the number the trained model must beat.
# ---------------------------------------------------------------------------
CODE_LOOKUP = {
    "BAD_REQUEST_ERROR": "insufficient_funds",
    "GATEWAY_ERROR": "bank_downtime",
    "SERVER_ERROR": "bank_downtime",
}

INTERVENTIONS = [
    "none",                 # do nothing (unrecoverable / opted out)
    "silent_retry",         # auto-retry, no customer contact
    "nudge_now",            # message immediately
    "nudge_delayed",        # message at the cause-appropriate time
    "new_instrument_link",  # payment link asking for a different method
    "incentive_link",       # payment link + capped discount
]
