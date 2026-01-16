import re
import json

def parse_medgemma_response(raw_text):
    """
    Parses the raw response from MedGemma.
    Extracts the thought process (between <unused94> and <unused95>)
    and the final JSON object.
    """
    # 1. Extract Thought Process (Using Regex)
    thought_match = re.search(r'<unused94>thought\s*(.*?)\s*<unused95>', raw_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None

    # 2. Extract JSON Data (Using Regex as requested)
    
    # Remove the thought part from the text to avoid matching braces inside thought
    text_without_thought = raw_text
    if thought_match:
        text_without_thought = raw_text.replace(thought_match.group(0), "")

    try:
        # We look for a JSON object structure: starting with { and ending with }
        # encompassing everything in between.
        json_match = re.search(r'(\{.*\})', text_without_thought, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
        else:
            data = None
            
    except json.JSONDecodeError:
        data = None
    
    # Fallback: Exception handling / malformed JSON
    # Return raw text if JSON parsing fails or no data
    if data is None:
        return text_without_thought.strip() if text_without_thought.strip() else raw_text

    return {"thought": thought, "data": data}
