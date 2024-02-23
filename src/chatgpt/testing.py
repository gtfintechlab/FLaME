import together



together.api_key = "1ba68d2ffcbdad1ac7dbc992797cfa0200a9031ab7c886e6701674892ba4acbf"


sentence = "however, following this deal, bce will divest about one-third of mts postpaid subscribers, for total proceeds of approximately $300 million, and 13 mts retail locations to its nearest national competitor, telus corporation to dispel regulatory concerns and trim cash outlay, as per prior agreement."

prompt = f'''Discard all the previous instructions. Behave like you are an expert sentence senti-
    ment classifier. Classify the following sentence into ‘INCLAIM’, or ‘OUTOFCLAIM’ class.
    Label ‘INCLAIM’ if consist of a claim and not just factual past or present information, or
    ‘OUTOFCLAIM’ if it has just factual past or present information. Provide the label in the
    first line and provide a short explanation in the second line. The sentence:{sentence}'''

output = together.Complete.create(
    prompt= f"<human>: {prompt} \n<bot>:",
    model="meta-llama/Llama-2-7b-chat-hf",
    max_tokens=256,
    temperature=0.8,
    top_k=60,
    top_p=0.6,
    repetition_penalty=1.1,
    stop=["<human>", "\n\n"],
)

# print generated text
print(output["output"]["choices"][0]["text"])
