from openai import AzureOpenAI
from prompt import prompt_top, prompt_10, random_prompt

client = AzureOpenAI(
  azure_endpoint="https://cs245canadaeast.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
  api_version="2024-08-01-preview",
  api_key="xxx"
)

prompt_names = ["top1", "top10", "random"]

for name in prompt_names:

    prompt = None
    if name == "top1":
        prompt = prompt_top
    elif name == "top10":
        prompt = prompt_10
    elif name == "random":
        prompt = random_prompt

    response = client.chat.completions.create(
        model="gpt-4o", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates training data. The format provided in the next prompt is perfect, ensure all 1000 question sets are created on the first run and exactly matches the sample output provided"},
            {"role": "user", "content": prompt}
        ]
    )

    with open(f"nq_{name}_own.txt", 'w') as f:
        f.write(response.choices[0].message.content)
