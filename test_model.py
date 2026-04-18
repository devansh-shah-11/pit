from make_paraphrase import make_story_by_calling_genai

prompt = "hello"
# history = [{
#         "role": "user",
#         "content": prompt
#     }]
history = [{'role': 'user', 'content': 'QUESTION: "Benny bought  2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?"\n\n            Instructions: Add noise to the QUESTION such that an llm solving this will get confused. Add random numbers that are not relevant. Make sure the real meaning and answer of the question does not change due to the noise. Answer ONLY the modified question.'}]

res = make_story_by_calling_genai("", history)

print("res")
print(res)

