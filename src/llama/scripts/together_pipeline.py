import together
from prompt_generator import *



def generate(task, model, api_key, sentence):
    
    
    
    together.api_key = api_key
    
    if (task == "numclaim"):
        prompt = numclaim_prompt(sentence)
        
    if (task == "fomc"):
        prompt = fomc_prompt(sentence)
    
    if (task == "finer"):
        prompt = finer_prompt(sentence)
    
    if (task == "fpb"):
        prompt = fpb_prompt(sentence)
        
    if (task == "finentity"):
        prompt = finentity_prompt(sentence)
        
    # if (task == "finsent"):
    #     prompt = finsent_prompt(sentence)
        
    if (task == "finqa"):
        prompt = finqa_prompt(sentence)
        
    if (task == "ectsum"):
        prompt = ectsum_prompt(sentence)
        
    # if (task == "finbench"):
    #     prompt = finbench_prompt(sentence)
        
    if (task == "banking77"):
        prompt = banking77_prompt(sentence)
        
    if (task == "convfinqa"):
        prompt = convfinqa_prompt(sentence)
    
    

    output = together.Complete.create(
        prompt= f"<human>: {prompt} \n<bot>:",
        model=model,
        max_tokens=256,
        temperature=0.8,
        top_k=60,
        top_p=0.6,
        repetition_penalty=1.1,
        stop=["<human>", "\n\n"],
    )

    return output