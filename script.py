from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from prompt_templates import intent_classification_prompt_template
from response_schemas import intent_classification_response_schema
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv
from dataset import pick_random_test
import json

load_dotenv()

# Initialize the language model
# Make sure you have set your OpenAI API key as an environment variable
llm = OpenAI(temperature=0.7)


# response schema parser
output_parser = StructuredOutputParser.from_response_schemas(intent_classification_response_schema)

# Get the format instructions
format_instructions = output_parser.get_format_instructions()


# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=intent_classification_prompt_template["input_variables"],
    template=intent_classification_prompt_template["template"] + "\n" +  "{format_instructions}",

)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt_template)

dataset = pick_random_test(n=5)
texts = dataset["sample"].tolist()

# predictions
results = []
for text in texts:


    inputs = {
        "input":text,
        "label":"intent",
        "targets":str(["good","bad"]),
        "format_instructions":format_instructions
    }
    # use the chain
    result = chain.run(inputs)
    parsed_output = output_parser.parse(result)
    parsed_output["text"] = text  
    results.append(parsed_output)

with open("results.json","w") as results_file:
    json.dump(results,results_file)
