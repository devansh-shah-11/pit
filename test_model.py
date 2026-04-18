from create_adverserial_dataset_test import ask_a_math_question


# prompt = "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. The truck also carries 13 red rubber gloves, 4 chrome-plated handcuffs, 9 silver flashlights, a bundle of 11 duct tapes, a lead pencil case, an ancient 19th-century cookbook, a half dozen spoons, and a 23‑inch telescope (though none of these are hard hats). If Carl removes 4 pink hard hats, and John removes 6 pink hard hats plus twice as many green hard hats as the number of pink hard hats he removed (and on a whim also snatches a random 5‑notebook file labeled “Project 42” that was perched on the corner of the cargo net), then calculate the total number of hard hats that remained in the truck."
# prompt = "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. The same truck also happens to be carrying 41 blue umbrellas, 3 gallon buckets of orange paint, 9 mismatched novelty coffee mugs, a bag of 17 sandbags, a 22‑inch kitesurfing board, 5 rubber chickens, and a single dusty harmonica (none of these items are hard hats). Carl seizes 4 pink hard hats, and John seizes 6 pink hard hats and twice as many green hard hats as the number of pink hard hats he seized (and in an inexplicable turn of events also tosses back a 12‑piece set of plastic measuring cups he had forgotten about). After all this, determine how many hard hats remain in the truck."
prompt = "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. The same truck also contains 9 brass candlesticks, 12 silver spatulas, a 7‑foot long rubber ladder, 4 clown shoes, 18 tangled fishing lines, a single pair of aviator sunglasses, and a 5‑layer chocolate cake (none of which are hard hats). Carl takes away 4 pink hard hats, while John takes away 6 pink hard hats and twice as many green hard hats as the pink hard hats he removed (plus an impulsive detour that collects 3 abandoned garden gnomes). Calculate the total number of hard hats that remain in the truck."
res = ask_a_math_question(prompt)
print("res:")
print(res)

# 0: 14 16 16 1
# 1: 10 8 20 12 8
# 2: 28 16 16 16 16

# 1: 46 2
# 2: 61 46
