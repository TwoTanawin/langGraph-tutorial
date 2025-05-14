from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrockConverse
import boto3

session = boto3.Session(profile_name="hydroneo", region_name="ap-southeast-1")
bedrock_client = session.client("bedrock-runtime")

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatBedrockConverse(
        model="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=bedrock_client,
        temperature=0.5,
    )

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm