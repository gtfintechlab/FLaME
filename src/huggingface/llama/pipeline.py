import enum
from typing import Dict, List

from transformers.pipelines import TextGenerationPipeline


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


from llama.instructions import E_INST, llama2_prompt_generator

from utils.logging import setup_logger

logger = setup_logger(__name__)


class LlamaTextGenerationPipeline(TextGenerationPipeline):
    def __init__(
        self,
        *args,
        task: str = "text-generation",
        # NOTE: Set `do_sample = True` when `temperature > 0.0`
        # https://github.com/huggingface/transformers/issues/25326
        temperature: float = 0.0,  # [0.0, 1.0]; 0.0 means greedy sampling
        do_sample: bool = False,
        max_new_tokens: int = 512,
        top_k: int = 10,
        top_p: float = 0.92,
        repetition_penalty: float = 1.0,  # 1.0 means no penalty
        num_return_sequences: int = 1,  # Only generate one response
        **kwargs,
    ):
        """
        # TODO: rewrite/update the docstring
        Initialize the TextGenerationPipeline for Llama2.
        Args:
            *args: Variable length argument list.
            task (str, optional): The task to use. Defaults to "text-generation".
            do_sample (bool, optional): Whether to use sampling. Defaults to True.
            max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 256.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
                probabilities that add up to top_p or higher are kept for generation. Defaults to 0.92.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Defaults to 0.
        """
        super().__init__(
            *args,
            task=task,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            **kwargs,
        )

    # TODO: Do not think we need _sanitize_parameters for Llama2
    # def _sanitize_parameters(self, return_full_text: bool = None, **generate_kwargs):
    #     preprocess_params = {}
    #
    #     # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
    #     # append a newline to yield a single token.  find whatever token is configured for the response key.
    #     tokenizer_response_key = next(
    #         (
    #             token
    #             for token in self.tokenizer.additional_special_tokens
    #             if token.startswith(RESPONSE_KEY)
    #         ),
    #         None,
    #     )
    #     response_key_token_id = None
    #     end_key_token_id = None
    #     if tokenizer_response_key:
    #         try:
    #             response_key_token_id = get_token_id(
    #                 self.tokenizer, tokenizer_response_key
    #             )
    #             end_key_token_id = get_token_id(self.tokenizer, END_KEY)
    #
    #             # Ensure generation stops once it generates "### End"
    #             generate_kwargs["eos_token_id"] = end_key_token_id
    #         except ValueError:
    #             pass
    #
    #     forward_params = generate_kwargs
    #     postprocess_params = {
    #         "response_key_token_id": response_key_token_id,
    #         "end_key_token_id": end_key_token_id,
    #     }
    #
    #     if return_full_text is not None:
    #         postprocess_params["return_full_text"] = return_full_text
    #
    #     return preprocess_params, forward_params, postprocess_params

    def preprocess(self, model_input, **generate_kwargs):
        instruction, sentence = model_input["instruction"], model_input["sentence"]
        prompt = llama2_prompt_generator(instruction, sentence)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        inputs["instruction"] = instruction
        inputs["sentence"] = sentence
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        # Generate text
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=(
                attention_mask.to(self.model.device)
                if attention_mask is not None
                else None
            ),
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        # Reshape the generated sequence to match the batch size of the input
        out_b = generated_sequence.shape[0]
        generated_sequence = generated_sequence.reshape(
            in_b, out_b // in_b, *generated_sequence.shape[1:]
        )
        instruction = model_inputs.pop("instruction")
        sentence = model_inputs.pop("sentence")
        return {
            "instruction": instruction,
            "sentence": sentence,
            "input_ids": input_ids,
            "generated_sequence": generated_sequence,
        }

    def postprocess(
        self,
        model_outputs,
        return_type: ReturnType = ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces: bool = True,
    ):
        instruction = model_outputs["instruction"]
        sentence = model_outputs["sentence"]
        generated_sequence: List[List[int]] = (
            model_outputs["generated_sequence"][0].numpy().tolist()
        )

        records = []
        for idx, sequence in enumerate(generated_sequence):
            # The response will be set to this variable if we can identify it.
            decoded = None

            # TODO: refactor the code below where we look for response/end keys for use in Llama2
            # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
            if E_INST:
                # Find `[/INST]` in the generated text and return the sequence found after the special token.
                try:
                    response_pos = sequence.index(E_INST)
                    decoded = self.tokenizer.decode(
                        sequence[response_pos + 1 :]
                    ).strip()
                except ValueError:
                    logger.error(f"Could not find response key {E_INST} in: {sequence}")
                    decoded = "ERROR: Could not find special token!"

            # If the full text is requested, then append the decoded text to the original instruction.
            # This technically isn't the full text, as we format the instruction in the prompt the model has been
            # trained on, but to the client it will appear to be the full text.
            # TODO: unsure ? remove all the special tokens when returning full text ?
            if return_type == ReturnType.FULL_TEXT:
                decoded = f"{instruction}\n{sentence}\n{decoded}"

            rec = {"generated_text": decoded}

            records.append(rec)

        return records
