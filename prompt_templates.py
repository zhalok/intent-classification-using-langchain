intent_classification_prompt_template = {
        "input_variables":["input","label","targets"],
        "template":'Classify the {label} of the text {input}. the all possible class labels are {targets}',
        "response_format":[
            {
                "class_label":"probability score of the class label prediction"
            }
        ]
    }





