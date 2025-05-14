from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
import boto3

session = boto3.Session(profile_name="hydroneo", region_name="ap-southeast-1")
bedrock_client = session.client("bedrock-runtime")

llm = ChatBedrockConverse(
        model="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=bedrock_client,
        temperature=0.5,
    )