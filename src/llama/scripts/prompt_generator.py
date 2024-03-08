def numclaim_prompt(sentence: str):
    
    prompt = f'''Discard all the previous instructions. Behave like you are an expert sentence senti-
            ment classifier. Classify the following sentence into ‘INCLAIM’, or ‘OUTOFCLAIM’ class.
            Label ‘INCLAIM’ if consist of a claim and not just factual past or present information, or
            ‘OUTOFCLAIM’ if it has just factual past or present information. Provide the label in the
            first line and provide a short explanation in the second line. The sentence:{sentence}'''
            
    return prompt

def fomc_prompt(sentence: str):
    prompt = f'''Discard all the previous instructions. Behave like you are an expert sentence clas-
                sifier. Classify the following sentence from FOMC into ‘HAWKISH’, ‘DOVISH’, or ‘NEU-
                TRAL’ class. Label ‘HAWKISH’ if it is corresponding to tightening of the monetary policy,
                ‘DOVISH’ if it is corresponding to easing of the monetary policy, or ‘NEUTRAL’ if the
                stance is neutral. Provide the label in the first line and provide a short explanation in the
                second line. The sentence: {sentence}'''
            
    return prompt


def finer_prompt(sentence: str):
    
    prompt = f'''Discard all the previous instructions. Behave like you are an expert named entity
                    identifier. Below a sentence is tokenized and each line contains a word token from the
                    sentence. Identify ‘Person’, ‘Location’, and ‘Organisation’ from them and label them. If the
                    entity is multi token use post-fix B for the first label and I for the remaining token labels
                    for that particular entity. The start of the separate entity should always use B post-fix for
                    the label. If the token doesn’t fit in any of those three categories or is not a named entity
                    label it ‘Other’. Do not combine words yourself. Use a colon to separate token and label.
                    So the format should be token:label. \n\n {{word tokens separated by \n/}}'''
            
    return prompt


def fpb_prompt(sentence: str):
    
    prompt = f'''Discard all the previous instructions. Behave like you are an expert sentence sentiment
                classifier. Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
                class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. Provide
                the label in the first line and provide a short explanation in the second line. The sentence:
                {sentence}'''
            
    return prompt


def finentity_prompt(sentence: str):
    
    prompt = f'''Discard all the previous instructions. Behave like you are an expert entity level sentiment
                classifier. Below is a sentence from a financial document. From the sentence, identify all the entities 
                check the starting and ending indices of the entities and give it a tag out of the following three options: 
                ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral.
                Format it as such: "start": start value, "end": end value, "value": entity name, 
                "tag":‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’. The sentence:
                {sentence}'''
            
    return prompt



# def fibench_prompt(sentence: str):
    
#     prompt = f'''Discard all the previous instructions. Behave like you are an expert entity level sentiment
#                 classifier. Below is a sentence from a financial document. From the sentence, identify all the entities 
#                 check the starting and ending indices of the entities and give it a tag out of the following three options: 
#                 ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
#                 corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral.
#                 Format it as such: "start": start value, "end": end value, "value": entity name, 
#                 "tag":‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’. The sentence:
#                 {sentence}'''
            
#     return prompt

def ectsum_prompt(document: str):
    
    prompt = f'''Discard all the previous instructions.
        Behave like you are an expert at summarization tasks.
        Below an earnings call transcript of a Russell 3000 Index company
        is provided. Perform extractive summarization followed by
        paraphrasing the transcript in bullet point format according to the
        experts-written short telegram-style bullet point summaries
        derived from corresponding Reuters articles. The target length of
        the summary should be at most 50 words. \n\n The document:
        {document}'''
            
    return prompt


banking77_list = ['activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support', 'automatic_top_up', 'balance_not_updated_after_bank_transfer', 'balance_not_updated_after_cheque_or_cash_deposit', 'beneficiary_not_allowed', 'cancel_transfer', 'card_about_to_expire', 'card_acceptance', 'card_arrival', 'card_delivery_estimate', 'card_linking', 'card_not_working', 'card_payment_fee_charged', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'card_swallowed', 'cash_withdrawal_charge', 'cash_withdrawal_not_recognised', 'change_pin', 'compromised_card', 'contactless_not_working', 'country_support', 'declined_card_payment', 'declined_cash_withdrawal', 'declined_transfer', 'direct_debit_payment_not_recognised', 'disposable_card_limits', 'edit_personal_details', 'exchange_charge', 'exchange_rate', 'exchange_via_app', 'extra_charge_on_statement', 'failed_transfer', 'fiat_currency_support', 'get_disposable_virtual_card', 'get_physical_card', 'getting_spare_card', 'getting_virtual_card', 'lost_or_stolen_card', 'lost_or_stolen_phone', 'order_physical_card', 'passcode_forgotten', 'pending_card_payment', 'pending_cash_withdrawal', 'pending_top_up', 'pending_transfer', 'pin_blocked', 'receiving_money', 'Refund_not_showing_up', 'request_refund', 'reverted_card_payment?', 'supported_cards_and_currencies', 'terminate_account', 'top_up_by_bank_transfer_charge', 'top_up_by_card_charge', 'top_up_by_cash_or_cheque', 'top_up_failed', 'top_up_limits', 'top_up_reverted', 'topping_up_by_card', 'transaction_charged_twice', 'transfer_fee_charged', 'transfer_into_account', 'transfer_not_received_by_recipient', 'transfer_timing', 'unable_to_verify_identity', 'verify_my_identity', 'verify_source_of_funds', 'verify_top_up', 'virtual_card_not_working', 'visa_or_mastercard', 'why_verify_identity', 'wrong_amount_of_cash_received', 'wrong_exchange_rate_for_cash_withdrawal']

def banking77_prompt(sentence: str):
    
    prompt = f'''Discard all the previous instructions. Behave like you are an expert at
                fine-grained single-domain intent detection. From the following list: {banking77_list}, identify
                which category does the following sentence belong to.
                {sentence}'''
            
    return prompt




