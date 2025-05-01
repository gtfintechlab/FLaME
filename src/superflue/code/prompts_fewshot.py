def headlines_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def fiqa_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def fiqa_task1_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def fiqa_task2_fewshot_prompt(question: str):
    # placeholder for the prompt
    return


def edtsum_fewshot_prompt(document: str):
    # placeholder for the prompt
    return


def numclaim_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def fomc_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def finer_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def fpb_fewshot_prompt(sentence: str, prompt_format: str):
    # placeholder for the prompt
    return


def finentity_fewshot_prompt(sentence: str):
    # placeholder for the prompt
    return


def finbench_fewshot_prompt(profile: str):
    # placeholder for the prompt
    return


def ectsum_fewshot_prompt(document: str):
    # placeholder for the prompt
    return


banking77_list = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def banking77_fewshot_prompt(sentence: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at
                fine-grained single-domain intent detection. From the following list: {banking77_list}, identify
                which category the following sentence belongs to.
                {sentence}"""
    return prompt


def finqa_fewshot_prompt(document: str):
    # placeholder for the prompt
    return


def convfinqa_fewshot_prompt(document: str):
    # placeholder for the prompt
    return


def tatqa_fewshot_prompt(document: str):
    # placeholder for the prompt
    return


def causal_classification_fewshot_prompt(text: str):
    # placeholder for the prompt
    return


def finred_fewshot_prompt(sentence: str, entity1: str, entity2: str):
    # placeholder for the prompt
    return


def causal_detection_fewshot_prompt(tokens: list):
    # placeholder for the prompt
    return


def subjectiveqa_fewshot_prompt(feature, definition, question, answer):
    # placeholder for the prompt
    return


def fnxl_fewshot_prompt(sentence, company, doc_type):
    # placeholder for the prompt
    return


def refind_fewshot_prompt(entities):
    # placeholder for the prompt
    return


prompt_map = {
    "numclaim_fewshot_prompt": numclaim_fewshot_prompt,
    "fiqa_task1_fewshot_prompt": fiqa_task1_fewshot_prompt,
    "fiqa_task2_fewshot_prompt": fiqa_task2_fewshot_prompt,
    "fomc_fewshot_prompt": fomc_fewshot_prompt,
    "finer_fewshot_prompt": finer_fewshot_prompt,
    "fpb_fewshot_prompt": fpb_fewshot_prompt,
    "finentity_fewshot_prompt": finentity_fewshot_prompt,
    "ectsum_fewshot_prompt": ectsum_fewshot_prompt,
    "edtsum_fewshot_prompt": edtsum_fewshot_prompt,
    "banking77_fewshot_prompt": banking77_fewshot_prompt,
    "finqa_fewshot_prompt": finqa_fewshot_prompt,
    "convfinqa_fewshot_prompt": convfinqa_fewshot_prompt,
    "tatqa_fewshot_prompt": tatqa_fewshot_prompt,
    "finred_fewshot_prompt": finred_fewshot_prompt,
    "causal_detection_fewshot_prompt": causal_detection_fewshot_prompt,
    "finbench_fewshot_prompt": finbench_fewshot_prompt,
    "refind_fewshot_prompt": refind_fewshot_prompt,
    "headlines_fewshot_prompt": headlines_fewshot_prompt,
    "fiqa_fewshot_prompt": fiqa_fewshot_prompt,
    "causal_classification_fewshot_prompt": causal_classification_fewshot_prompt,
    "subjectiveqa_fewshot_prompt": subjectiveqa_fewshot_prompt,
    "fnxl_fewshot_prompt": fnxl_fewshot_prompt,
}


def prompt_function(prompt_name):
    return prompt_map.get(prompt_name, None)
