# summary_script.py
import os
import json
import boto3
from datetime import datetime
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
import logging
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
logging.getLogger('pymongo.connection').setLevel(logging.WARNING)
logging.getLogger('pymongo.serverSelection').setLevel(logging.WARNING)
# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize MongoDB
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["test"]
summary_collection = mongo_db["summaries"]

# Initialize S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def generate_summary(agent_id: str, call_id: str, transcript: str, user_info: dict, db):
    """
    Generate a structured summary JSON using AI and Firestore knowledge base.
    """
    print(f"Generating summary for agent {agent_id}, call {call_id}")
    
    contact_info = {
    "email": user_info.get("email", "Unknown"),
    "phone": user_info.get("phone", "None")
}

    # Use the passed db client
    kb_ref = db.collection("agents").document(agent_id).collection("knowledge_base")
    docs = kb_ref.stream()
    knowledge_base = [doc.to_dict() for doc in docs]
    knowledge_text = " ".join([item.get("content", "") for item in knowledge_base])

    # Use AI to generate summary
    prompt = f"""
    You are an intelligent assistant. A customer just finished a conversation with an AI agent.
    Here is the full transcript:
    {transcript}

    Agent knowledge base:
    {knowledge_text}
    
    IMPORTANT: You MUST return a JSON object with EXACTLY this structure:
{{
  "requestedData": "string describing what the user requested",
  "responseData": "string with the response/answer provided",
  "contactInfo": {{
    "email": "extract from transcript or 'Unknown'",
    "phone": "extract from transcript or 'None'"
  }},
  "deliveryChannels": ["array", "of", "strings"]
}}

    Based on the conversation, identify:
    1. What the user requested (requestedData)
    2. The most relevant response (responseData) — using the knowledge base above if possible.
    3. The user's contact info: {json.dumps(contact_info, indent=2)}
    4. Delivery channels (email, whatsapp) — inferred if mentioned.

    Return a valid JSON with:
    requestedData, responseData, contactInfo, and deliveryChannels.
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates structured JSON summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        response_format={"type": "json_object"}
    )

    ai_output = completion.choices[0].message.content

    # Parse AI JSON safely
    try:
        summary_data = json.loads(ai_output)
    except json.JSONDecodeError:
        summary_data = {
            "requestedData": "N/A",
            "responseData": "Unable to parse AI response.",
            "contactInfo": contact_info,
            "deliveryChannels": ["email"]
        }

    # Upload summary to S3
    file_name = f"summary_{agent_id}_{call_id}.json"
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=file_name,
        Body=json.dumps(summary_data, indent=2),
        ContentType="application/json"
    )
    s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}"

    # Save metadata to MongoDB
    summary_collection.insert_one({
        "agentId": agent_id,
        "callId": call_id,
        "summaryUrl": s3_url,
        "createdAt": datetime.utcnow()
    })

    print(f"✅ Summary generated and stored at {s3_url}")
    return summary_data







