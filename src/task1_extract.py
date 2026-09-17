import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PDF_FILE = "SECTION_071400_Bituminous_Waterproofing_System.pdf"
OUTPUT_FILE = "my_model_output.json"

# OpenAI currently recommends the Responses API for new text-generation work.
# GPT-5.6 Terra gives a balance between reasoning quality and cost.
MODEL_NAME = "gpt-5.6-terra"


# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------

# Read variables stored in the .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Check that your .env file contains the API key."
    )

# Create the OpenAI client
client = OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Read text from the PDF
# ---------------------------------------------------------

def extract_pdf_text(pdf_path):
    """
    Read all pages from a PDF and combine them into one text string.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Keep page information because section/page context
        # can help the model understand where statements appear.
        if text:
            pages.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

    return "\n".join(pages)


document_text = extract_pdf_text(PDF_FILE)

print("PDF loaded successfully.")
print("Number of characters extracted:", len(document_text))


# ---------------------------------------------------------
# Build the instruction for the LLM
# ---------------------------------------------------------

prompt = f"""
You are reviewing a construction specification for internal contradictions.

Your job is to identify genuine contradictions within the document.

Focus on these four types of contradiction:

1. Technical
   Example: two sections specify incompatible material dimensions,
   strengths, temperatures, or installation requirements.

2. Responsibility
   Example: two different parties are both assigned sole responsibility
   for the same work.

3. Liability
   Example: two clauses assign the same liability or cost to different
   parties in incompatible ways.

4. Schedule
   Example: two clauses require incompatible waiting periods,
   curing periods, or timing requirements.

Important rules:

- Compare statements across the whole document.
- Do not use exact wording as the test for a contradiction.
- A contradiction exists when two requirements cannot reasonably
  be followed at the same time.
- Do not report two statements merely because they mention
  different numbers if the numbers refer to different things.
- Do not report statements that are simply repeated.
- Do not report normal clarifications or additional requirements
  as contradictions.
- Do not invent contradictions just to reach a particular number.
- Prefer clear, defensible conflicts.


Important:

A conflict does not need to be logically impossible in the strictest sense.

Also report inconsistent requirements where:
- two sections give different minimum values for the same requirement
- two sections give different warranty periods
- two different parties are given final approval authority
- one clause gives a general rule while another gives a conflicting exception
- two clauses create ambiguity about which requirement should govern

Compare statements across the whole document.

Do not report simple repetition.

Do not report unrelated numbers that refer to different things.

Do not invent conflicts just to reach a certain number.

Before returning the final list, remove duplicate contradictions that describe the same underlying conflict.

Also check carefully for:
- general rules versus conditional exceptions
- different curing periods or waiting periods
- conflicting schedule requirements

Return ONLY valid JSON.


Use exactly this structure:

{{
    "document": "Section 071400 - Bituminous Waterproofing System",
    "total_contradictions": 0,
    "contradictions": [
        {{
            "Statement1": "First conflicting statement",
            "Statement2": "Second conflicting statement",
            "Reasoning": "Short explanation of why these two statements conflict"
        }}
    ]
}}

Set "total_contradictions" to the actual number of contradictions you find.

Here is the specification:

{document_text}
"""


# ---------------------------------------------------------
# Ask the LLM to find contradictions
# ---------------------------------------------------------

print("\nSending document to the model...")

response = client.responses.create(
    model=MODEL_NAME,
    input=prompt
)

# The Responses API provides the generated text through output_text
result_text = response.output_text

print("Model response received.")


# ---------------------------------------------------------
# Convert the model response into JSON
# ---------------------------------------------------------

try:
    result_json = json.loads(result_text)

except json.JSONDecodeError:
    print("\nThe model response was not valid JSON.")
    print("Raw response:\n")
    print(result_text)
    raise


# ---------------------------------------------------------
# Save Task 1 output
# ---------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        result_json,
        file,
        indent=2,
        ensure_ascii=False
    )


print(f"\nTask 1 complete.")
print(f"Contradictions found: {result_json['total_contradictions']}")
print(f"Output saved to: {OUTPUT_FILE}")