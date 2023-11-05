import transformers
from transformers import pipeline

# Load the pre-trained BERT model
model_name = 'bert-base-uncased'
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Define a function to generate a summary or answer questions based on the content
def generate_summary_or_answer(text, question=None, max_length=512):
    if question:
        inputs = tokenizer.encode_plus(question, text, return_tensors='pt', max_length=max_length, truncation=True)
        input_ids = inputs['input_ids'].tolist()[0]
        answer_start_scores, answer_end_scores = model(**inputs)
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer
    else:
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']

# Example usage
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam dapibus magna ut justo eleifend, id faucibus mauris semper. Sed in semper dolor. Sed nec ipsum ut lorem laoreet vehicula. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Curabitur non semper ligula. Duis fermentum, tellus id congue feugiat, elit arcu mattis enim, a aliquet metus mauris vitae enim. Sed auctor, justo ut tempor fringilla, nisi turpis tincidunt neque, vitae malesuada ligula metus vitae tortor. Fusce nec purus non turpis tincidunt lacinia. Sed consectetur, tellus at malesuada euismod, odio arcu dictum urna, eget commodo purus risus nec augue. Sed sed congue mi. Sed varius, dui sit amet pellentesque porta, risus nunc vehicula odio, ut ultricies quam justo nec metus. Donec et elit id lorem fringilla auctor. Nunc ut sollicitudin elit, sed pharetra lorem. Sed non mauris interdum, mattis lectus nec, scelerisque diam. Nam vel laoreet felis, id efficitur tellus. Vivamus finibus, enim id congue porttitor, justo ligula malesuada nisi, in lacinia metus tellus eu ligula."
question = "What is Lorem ipsum?"

summary = generate_summary_or_answer(text)
answer = generate_summary_or_answer(text, question)

print("Summary:")
print(summary)
print("\nAnswer to the question:")
print(answer)
