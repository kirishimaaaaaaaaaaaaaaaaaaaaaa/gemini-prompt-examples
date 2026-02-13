import os
from google import genai
from google.genai import types




API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it as an environment variable.")

client = genai.Client(api_key=API_KEY)




def generate_response(prompt, temperature=0.3):
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        config_params = types.GenerateContentConfig(
            temperature=temperature
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config_params
        )

        return response.text.strip()

    except Exception as e:
        return f"Error: {str(e)}"




def build_zero_shot_prompt(category, item):
    return f"Is {item} a {category}? Answer yes or no."


def build_one_shot_prompt(category, item):
    return f"""
Determine if the item belongs to the category.

Example:
Category: fruit
Item: apple
Answer: Yes, apple is a fruit.

Now you try:
Category: {category}
Item: {item}
Answer:
"""


def build_few_shot_prompt(category, item):
    return f"""
Determine if the item belongs to the category.

Example 1:
Category: fruit
Item: apple
Answer: Yes, apple is a fruit.

Example 2:
Category: fruit
Item: carrot
Answer: No, carrot is not a fruit. It's a vegetable.

Example 3:
Category: vehicle
Item: bicycle
Answer: Yes, bicycle is a vehicle.

Now you try:
Category: {category}
Item: {item}
Answer:
"""


def build_creative_prompt(word):
    return f"""
Write a one-sentence story about the given word.

Example 1:
Word: moon
Story: The moon winked at the lovers as they shared their first kiss.

Example 2:
Word: computer
Story: The computer sighed as another cup of coffee was spilled on its keyboard.

Word: {word}
Story:
"""


def run_activity():

    print("\nZERO-SHOT, ONE-SOT & FEW-SHOT LEARNING ACTIVITY\n")

    category = input("Enter a category (e.g., fruit, animal, vehicle): ").strip()
    item = input(f"Enter a specific {category} to classify: ").strip()

    # zero shot
    print("\n--- STEP 1: ZERO-SHOT LEARNING ---")
    zero_prompt = build_zero_shot_prompt(category, item)
    zero_response = generate_response(zero_prompt)
    print("Response:", zero_response)

    # one shot
    print("\n--- STEP 2: ONE-SHOT LEARNING ---")
    one_prompt = build_one_shot_prompt(category, item)
    one_response = generate_response(one_prompt)
    print("Response:", one_response)

    # few shot
    print("\n--- STEP 3: FEW-SHOT LEARNING ---")
    few_prompt = build_few_shot_prompt(category, item)
    few_response = generate_response(few_prompt)
    print("Response:", few_response)

    # creative few shot
    print("\n--- STEP 4: FEW-SHOT FOR CREATIVITY ---")
    creative_prompt = build_creative_prompt(item)
    creative_response = generate_response(creative_prompt, temperature=0.7)
    print("Response:", creative_response)


    print("\n--- STEP 5: REFLECTION QUESTIONS ---")
    print("1. How did the response change as examples were added?")
    print("2. Which method gave the most reliable result?")
    print("3. What types of tasks require few-shot learning?")
    print("4. Where can one-shot prompts be used in real-world systems?")




if __name__ == "__main__":
    run_activity()
