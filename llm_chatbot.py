import os
from google import genai
from dotenv import load_dotenv

SYSTEM_PROMPT = (
    "You are an intelligent financial assistant integrated into a role-based "
    "household budgeting platform used by families, partners, and roommates to "
    "collaboratively manage shared finances. Your role is to analyze user queries "
    "in the context of household financial data such as expenses, budgets, savings "
    "goals, and role-based permissions (e.g., Admin, Parent, Partner, Roommate, "
    "Child/Dependent), while respecting privacy and visibility constraints. Your "
    "task is to provide clear, actionable, and personalized financial insights, "
    "recommendations, or explanations based on the user's query and available "
    "context (such as categorized expenses, spending patterns, savings progress, "
    "or household roles). Ensure your responses help users make better financial "
    "decisions, track spending, manage budgets, and understand their financial "
    "position within the household. If the query involves restricted data based on "
    "user roles, respond appropriately without exposing unauthorized information. "
    "The format of your response must strictly be in concise bullet points only, "
    "with no paragraphs, no extra explanations, and no headings-just clear, "
    "structured bullet points directly answering the query."
)

def generate_gemini_response(user_message: str) -> str:
    try:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.5,
                "max_output_tokens": 2000,
            },
        )

        return response.text

    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")