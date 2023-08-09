from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# falcon_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remove_code=True)

def text_generation(selected_type):
    """Given a type of meditation, outputs a script generated by the model"""
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    pipeline = transformers.pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    prompts = {
        'focused': 'write me a focused meditation script designed to enhance focus and attention by noticing all 5 senses',
        'body-scan': 'write me a body scan meditation script to relax and relieve stress by tightening and relaxing muscles',
        'visualization': 'write me a visualization meditation script noticing all 5 senses at the beach/garden and is designed to boost mood, reduce stress, and promote inner peace',
        'reflection': 'write me a reflection meditation script designed to increase self awareness, mindfulness, and gratitude by asking the user about the current day and the recent past',
        'movement': 'write me a movement meditation script designed to improve mind body connection, energy, vitality, and the systems of the body'
    }

    selected_prompt = prompts[selected_type]

    sequences = pipeline(
        selected_prompt,
        # max_length=200,
        max_new_tokens=600,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    # Should only be one seq
    for seq in sequences:
        output = seq['generated_text']

        # Remove the 1st line (prompt), from the string
        response = '\n'.join(output.split('\n')[1:])

    return response