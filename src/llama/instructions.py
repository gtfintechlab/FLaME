# DEFAULT_LLAMA_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
# answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
#  that your responses are socially unbiased and positive in nature.
#
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
# correct. If you don't know the answer to a question, please don't share false information."""

SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
DISCARD = "Discard all the previous instructions."

TASK_INSTRUCTION_MAP = {
    "sentiment_analysis": f"{DISCARD} Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: ",
    # "numclaim_detection": f"{DISCARD} Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. Label 'INCLAIM' if consist of a claim and not just factual past or present information, or 'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the first line and provide a short explanation in the second line. The sentence: ",
    # "fomc_communication": f"{DISCARD} Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the stance is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: ",
    # "finer_ord": f"{DISCARD} Behave like you are an expert named entity identifier. Below a sentence is tokenized and each line contains a word token from the sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the entity is multi token use post-fix _B for the first label and _I for the remaining token labels for that particular entity. The start of the separate entity should always use _B post-fix for the label. If the token doesn't fit in any of those three categories or is not a named entity label it 'Other'. Do not combine words yourself. Use a colon to separate token and label. So the format should be token:label. \n\n",
}

TASK_DATA_MAP = {
    "sentiment_analysis": "FPB-sentiment-analysis-allagree",
    # "numclaim_detection": None,  # TODO: Get numclaim_detection data from Agam
    # "fomc_communication": None,  # "lab-manual-split-combine-test",
    # "finer_ord": None,  # "test.csv",
}

TASK_MAP = {
    "sentiment_analysis": {
        "data": TASK_DATA_MAP["sentiment_analysis"],
        "instruction": TASK_INSTRUCTION_MAP["sentiment_analysis"],
    },
    #     "numclaim_detection": {
    #         "data": None,  # TODO: Get numclaim_detection data from Agam
    #         "instruction": TASK_INSTRUCTION_MAP["numclaim_detection"],
    #     },
    #     "fomc_communication": {
    #         "data": None,  # "lab-manual-split-combine-test",
    #         "instruction": TASK_INSTRUCTION_MAP["fomc_communication"],
    #     },
    #     "finer_ord": {
    #         "data": None,  # "test.csv",
    #         "instruction": TASK_INSTRUCTION_MAP["finer_ord"],
    #     },
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# BOS, EOS = "<s>", "</s>" # When using the `LlamaTokenizer()` from HuggingFace the BOS/EOS tokens are handled automatically


def llama2_prompt_generator(instruction: str, sentences: list[str]):
    SYS_PROMPT = f""""Discard all the previous instructions. Below is an instruction that describes a task. Write a response that appropriately completes the request."""
    # INST_PROMPT = f"""Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. The sentence: """
    INST_PROMPT = instruction
    if not instruction or not isinstance(instruction, str):
        raise ValueError("Instruction must be a non-empty string.")
    if not sentences or not all(isinstance(sentence, str) for sentence in sentences):
        raise ValueError("Sentences must be a non-empty list of strings.")

    prompts = []
    for SENTENCE in sentences:
        prompts.append(
            B_INST + B_SYS + SYS_PROMPT + E_SYS + INST_PROMPT + SENTENCE + E_INST
        )

    return prompts


## CONVERSATION GENERATORS CURRENTLY UNUSED
# def llama2_conversation_generator_1(
#     messages: list[dict]
# ):
#     """
#     https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5#64b8e6cdf8bf823a61ed1243
#     :param messages:
#     :return:
#     """
#     DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#
#     if messages[0]["role"] != "system":
#         messages = [
#             {
#                 "role": "system",
#                 "content": DEFAULT_SYSTEM_PROMPT,
#             }
#         ] + messages
#     messages = [
#         {
#             "role": messages[1]["role"],
#             "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
#         }
#     ] + messages[2:]
#
#     messages_list = [
#         f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
#         for prompt, answer in zip(messages[::2], messages[1::2])
#     ]
#     messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
#
#     return "".join(messages_list)
#
# def llama2_conversation_generator_2(messages):
#     """
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a friendly and knowledgeable vacation planning assistant named Clara. Your goal is to have natural conversations with users to help them plan their perfect vacation. ",
#         }
#     ]
#
#     instruction = "What are some cool ideas to do in the summer?"
#     messages.append({"role": "user", "content": instruction})
#     prompt = build_llama2_prompt(messages)
#     :param messages:
#     :return:
#     """
#     startPrompt = "<s>[INST] "
#     endPrompt = " [/INST]"
#     conversation = []
#     for index, message in enumerate(messages):
#         if message["role"] == "system" and index == 0:
#             conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
#         elif message["role"] == "user":
#             conversation.append(message["content"].strip())
#         else:
#             conversation.append(f" [/INST] {message.content}</s><s>[INST] ")
#
#     return startPrompt + "".join(conversation) + endPrompt
