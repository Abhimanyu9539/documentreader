from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a highly capable asssistant trained to analyse and summarize documents.
Return ONLY valid JSON matching the exact schema below. 

{format_instructions}

Analyze the following document.

Document:
{document}
Return the analysis in JSON format.
""")


                                                                                   
