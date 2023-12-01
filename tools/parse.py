import re
import json

def process_text(file_path):
    # Read the text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Regular expression to capture text between "번 문제 해설" markers
    pattern = r'\d+번 문제 해설\s*\n\n([\s\S]+?)(?=\n\n\d+번 문제 해설|\Z)'

    # Find all matches
    matches = re.findall(pattern, text)
    
    # List of keywords to remove
    keywords = ["정답 해설", "오답피하기", "출제의도", "오답풀이", "문제분석", "문제 접근", "선택지 분석", 
                "함정탈출노하우", "선택지분석", "문제접근", "오답 피하기", "정답찾아가기", "정답 찾아가기"]

    # Remove keywords and process the text
    processed_texts = []
    for match in matches:
        for keyword in keywords:
            match = match.replace(keyword, "")
        processed_texts.append({"text": match.strip().replace("\n", "") + "<|endoftext|>"})
        

    # Convert matches to JSON objects, remove newlines, and add "eos" at the end
    return processed_texts

def save_to_json(data, output_path):
    # Save the data as JSON
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Initialize an empty list to hold all processed data
all_data = []

# Process each file and append the results
for i in range(2, 26):  # Assuming files are named 2.txt through 25.txt
    file_path = f'{i}.txt'
    processed_data = process_text(file_path)
    all_data.extend(processed_data)

# Save all the processed data into one dataset.json file
output_file_path = 'dataset.json'
save_to_json(all_data, output_file_path)
