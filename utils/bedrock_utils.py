import json
import sys
import re

def create_llama3_body(messages, max_gen_len=2048, temperature=0.0, top_p=0.9, top_k=1):
    """
    Create a request body for Llama 3 models.

    Args:
    messages (list): List of message dictionaries.
    max_gen_len (int): Maximum generation length.
    temperature (float): Temperature for sampling.
    top_p (float): Top-p sampling parameter.
    top_k (int): Top-k sampling parameter.

    Returns:
    str: JSON-encoded string representing the request body for Llama 3 models.

    This function formats the input messages into a prompt suitable for Llama 3 models,
    including specific formatting tags, and creates a structured request body.
    """
    prompt = "\n".join([content for message in messages for content in message["content"]])
    formatted_prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {prompt.strip()}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    return json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "top_p": top_p,
    })


def extract_innermost_text(content):
    """Recursively extract the innermost `text` value from a deeply nested structure."""
    if isinstance(content, list) and content:  # If content is a list, dive into the first element
        return extract_innermost_text(content[0])
    elif isinstance(content, dict) and "text" in content:  # If content is a dictionary, dive into the 'text' key
        return extract_innermost_text(content["text"])
    elif isinstance(content, str):  # Base case: return the string when reached
        return content
    return ""  # Fallback in case the structure is invalid

def create_nova_messages(messages):
    """
    Create messages array for Nova models from conversation

    Args:
    conv (object): Conversation object containing messages

    Returns:
    list: List of formatted messages for Nova model
    """
    messages_formatted = []    
    # Format the first message with template
    for mesg in messages:        
        # Transform the message
        transformed_message = {
            "role": mesg["role"],
            "content": extract_innermost_text(mesg["content"])
        }
        messages_formatted.append({
            "role": "user",
            "content": [
                { 
                    "text": transformed_message['content']
                }
            ]
        })    
    return messages_formatted

def extract_answer(text):
    """
    Extract the content after the </think> tag.
    
    Args:
        text (str): Input text that may contain a </think> tag
        
    Returns:
        str: Text after the </think> tag, or the original text if tag not found
    """
    # Look for the </think> tag
    match = re.search(r'</think>(.*)', text, re.DOTALL)
    
    if match:
        # Return only the content after the tag
        return match.group(1).strip()
    else:
        # If tag not found, return empty string
        return ""

