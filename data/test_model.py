import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.make_paraphrase import make_story_by_calling_genai

prompt = "hello"
# history = [{
#         "role": "user",
#         "content": prompt
#     }]
history = [{'role': 'user', 'content': 'QUESTION: "Benny bought  2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?"\n\n            Instructions: Add noise to the QUESTION such that an llm solving this will get confused. Add random numbers that are not relevant. Make sure the real meaning and answer of the question does not change due to the noise. Answer ONLY the modified question.'}]

start_time = time.time()
res = make_story_by_calling_genai("", history)
print("TIME", time.time() - start_time)

print("res")
print(res)

