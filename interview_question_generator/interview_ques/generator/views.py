import os
import pandas as pd
import torch
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from difflib import get_close_matches

# === Load and Prepare CSV ===
CSV_PATH = os.path.join(settings.BASE_DIR, "generator", "static", "cleaned_interview_questions.csv")
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.lower()
required_columns = {"category", "difficulty", "question"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"❌ CSV is missing required columns: {df.columns.tolist()}")

# === Load GPT-2 Model ===
MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# === Utility: Fuzzy Match Category ===
def fuzzy_match_category(category_input):
    """Fuzzy match user-entered category to known categories."""
    all_categories = df["category"].dropna().str.lower().unique().tolist()
    match = get_close_matches(category_input.lower(), all_categories, n=1, cutoff=0.5)
    return match[0] if match else category_input

# === Utility: Get Questions from CSV ===
def get_questions_from_csv(category):
    """Fetch questions from CSV based on (fuzzy-matched) category."""
    filtered_df = df[df["category"].str.contains(category, case=False, na=False, regex=True)]

    if filtered_df.empty:
        return None

    easy_q = filtered_df[filtered_df["difficulty"].str.lower() == "easy"]["question"].dropna().tolist()
    medium_q = filtered_df[filtered_df["difficulty"].str.lower() == "medium"]["question"].dropna().tolist()
    hard_q = filtered_df[filtered_df["difficulty"].str.lower() == "hard"]["question"].dropna().tolist()

    if not (easy_q or medium_q or hard_q):
        return None

    return {
        "Easy": easy_q,
        "Medium": medium_q,
        "Hard": hard_q,
    }

# === Utility: Generate AI Question ===
def generate_with_gpt(category):
    """Generate an AI-based question using GPT-2."""
    prompt = f"Generate a unique interview question related to {category}:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text.replace(prompt, "").strip()

    # Return only the first complete sentence (question)
    if "?" in generated_text:
        generated_text = generated_text.split("?")[0] + "?"

    return generated_text

# === Main View Function ===
def generate_question(request):
    """Handles AJAX request for question generation."""
    if request.method == "POST":
        category_input = request.POST.get("category", "").strip()

        if not category_input:
            print("🚨 Error: Category is empty!")
            return JsonResponse({"error": "Category cannot be empty."}, status=400)

        # Fuzzy match user input
        matched_category = fuzzy_match_category(category_input)
        print(f"🔎 User Input: '{category_input}' | Matched: '{matched_category}'")

        # Try getting questions from CSV
        questions = get_questions_from_csv(matched_category)
        print(f"🗂️ CSV Questions Found: {bool(questions)}")

        if questions:
            return JsonResponse({"questions": questions})
        else:
            try:
                print(f"🤖 Generating with GPT for: '{matched_category}'")
                gpt_question = generate_with_gpt(matched_category)
                print(f"✅ GPT Output: {gpt_question}")
                return JsonResponse({"gpt_question": gpt_question})
            except Exception as e:
                print(f"❌ GPT Generation Error: {e}")
                return JsonResponse({"error": "Failed to generate a question."}, status=500)

    return render(request, "generator/form.html")
