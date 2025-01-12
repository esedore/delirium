# Model Confustion Guide

## Methods & Rationale
Currently this project aims to research methods which can spiral text to image AI models into a state of confusion & collapse.
The dataset of good text to image prompts used for "prompts.csv" comes from *[Krea-ai](https://docs.google.com/uc?export=download&id=1c4WHxtlzvHYd0UY5WCMJNn2EO-Aiv2A0)*

### Input generation: 
Develop a script or program that generates repetitive input patterns to be sent to the model. This script can generate similar keywords, attributes, or other relevant input parameters. Make sure to randomize certain aspects to mimic real-world variations.

### Feedback generation: 
Similarly, create a script that generates incorrect or misleading feedback for the model. This script can provide feedback that contradicts the expected output or reinforces incorrect image features. Again, randomness can be introduced to simulate different types of incorrect feedback.

### Test case execution: 
Automate the execution of the generated input and feedback scripts against the target model. This can be done by integrating with the model's API or using appropriate web scraping techniques to interact with the front-end interface.

### Output analysis: 
Develop mechanisms to capture and analyze the model's responses. This may involve extracting the generated images or analyzing the output logs to detect patterns or issues related to model collapse or incorrect handling of feedback.

### Reporting: 
Create an automated reporting system that consolidates the findings from the tests. This system should generate clear and detailed reports highlighting any vulnerabilities or issues encountered during the automated testing process.

### Iterative refinement: 
Continuously refine and improve your automation scripts based on the insights gained from previous test runs. This can involve adjusting the input generation logic, expanding the range of repetitive patterns, or refining the feedback generation process.

### Test Attack GenAI - Webby.py
https://delirium-3bqff74usa9zwh3a67cjq2.streamlit.app/
