import openai
import json

def generate_instruction_and_input(text, model="gpt-3.5-turbo", temperature=0.7, max_tokens=2000):
    """
    Generate an instruction and input for the given text using OpenAI API.

    Parameters:
    text (str): The output text.
    model (str): The GPT model to use.
    temperature (float): The randomness in response.
    max_tokens (int): Maximum length of the response.

    Returns:
    dict: A dictionary containing the instruction and input.
    """
    openai.api_key = ''  # Replace with your API key

    # Example prompt: "Create a learning objective and input for the following educational content: {text}"
    prompt = f"""Create an
    instruction and input for the following content: {text}
    every content must be Korean.
    response format is like this
    ---
    instruction: "병인양요가 흥선 대원군의 대외 정책을 어떻게 반영하는지 설명하세요."
    input: "병인양요 중 흥선 대원군의 대응."
    
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    # Remove "instruction: " and "input: " from the response
    cleaned_text = response.choices[0].message['content'].strip().replace("instruction: ", "").replace("input: ", "")
    generated_text = cleaned_text.split("\n")

    # Check if the generated text has at least two lines
    if len(generated_text) >= 2:
        # First line is the instruction, second line is the input
        return {"instruction": generated_text[0], "input": generated_text[1], "output": text}
    else:
        # If not, treat the entire cleaned response as the instruction
        return {"instruction": cleaned_text, "input": "", "output": text}
# Load your dataset
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Process in batches
batch_size = 10
new_data = []

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    for item in batch:
        result = generate_instruction_and_input(item['text'])
        new_data.append(result)
        print("Processed item:", result)  # Optional, for progress tracking

    # Save the new data after each batch
    with open('new_dataset.json', 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)
    print(f"Batch {i//batch_size + 1} processed and saved.")

print("Processing complete. New dataset saved as 'new_dataset.json'.")
