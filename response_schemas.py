from langchain.output_parsers import  ResponseSchema

# schema for the response of the llm
intent_classification_response_schema =[
    ResponseSchema(name="class_label", description="The name of the class label"),
    ResponseSchema(name="score", description="Confidence score of the class label prediction"),
   
]
