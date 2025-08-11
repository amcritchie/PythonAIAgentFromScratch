from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

# Parse the nested JSON structure
try:
    # The output is a string containing JSON wrapped in markdown code blocks
    output_text = raw_response.get("output")
    
    # Extract JSON from markdown code blocks if present
    if output_text and "```json" in output_text:
        # Find the JSON content between ```json and ```
        start_idx = output_text.find("```json") + 7
        end_idx = output_text.rfind("```")
        if start_idx > 6 and end_idx > start_idx:
            json_content = output_text[start_idx:end_idx].strip()
            print("Extracted JSON content:")
            print(json_content)
            
            # Parse the extracted JSON
            import json
            parsed_data = json.loads(json_content)
            print("-------------------------------4")
            print("Parsed data:")
            print(f"Topic: {parsed_data.get('topic')}")
            print(f"Summary: {parsed_data.get('summary')}")
            print(f"Sources: {parsed_data.get('sources')}")
            print(f"Tools used: {parsed_data.get('tools_used')}")
            
            # Also try to parse with Pydantic if the structure matches
            try:
                structured_response = parser.parse(json_content)
                print("-------------------------------5")
                print("Pydantic parsed response:")
                print(structured_response)
            except Exception as pydantic_error:
                print(f"Pydantic parsing failed: {pydantic_error}")
        else:
            print("Could not extract JSON from markdown blocks")
    else:
        print("No JSON markdown blocks found in output")
        print("Raw output:", output_text)
        
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)