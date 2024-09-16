# generation_haiku.py

import json
import os
import random
import re
from typing import Any, List, Union

from datasets import Dataset
from dotenv import load_dotenv

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import SelfInstructTask

# Load environment variables
load_dotenv()


class OpenAILLM(AsyncLLM):
    def __init__(
        self,
        model: str,
        api_base: str = None,
        api_key: str = None,
        max_retries: int = 6,
        timeout: int = 120,
        task: Any = None,
        max_tokens: int = 128,
        temperature: float = 0.4,
    ):
        super().__init__()
        self.model = model
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.task = task
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def agenerate(
        self, input: str, num_generations: int = 1, **kwargs: Any
    ) -> List[Union[str, None]]:
        import openai

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.task.system_prompt},
                    {"role": "user", "content": input},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=num_generations,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            print(f"Error generating response: {e}")
            return [None] * num_generations


# Function to stream content from original_dataset.jsonl
def stream_jsonl(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield json.loads(line)


# Preestablished prompts
preestablished_prompts = [
    "Write a haiku about {}",
    "Compose a haiku inspired by {}",
    "Create a haiku that captures the essence of {}",
    "Craft a haiku reflecting on {}",
    "Pen a haiku that explores the theme of {}",
]

# Define application description and criteria for query generation
application_description = (
    "An AI assistant adept at writing Haiku. "
    "It expects complete suggestions from users providing details of the kind of haiku they want. "
    "The AI assistant will help users write haiku about particular topics and is willing to accept requests related to a specific subject or object or a more abstract request "
    "based on an emotion, theme or vibe."
)

criteria_queries = (
    "Incorporate a diverse range of verbs, avoiding repetition.\n"
    "Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.\n"
    "Design queries to be self-contained and standalone."
)

# Create the instruction task
instruction_task = SelfInstructTask(
    system_prompt="You are an expert Haiku writer, writing the best and most diverse Haiku given topics as inputs.",
    application_description=application_description,
    criteria_for_query_generation=criteria_queries,
    num_instructions=15,
)

# Create the LLM (using OpenAI API compatible endpoint)
llm = OpenAILLM(
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    task=instruction_task,
    max_tokens=128,
    temperature=0.4,
)

# Create the pipeline
pipeline = Pipeline(generator=llm)


# Function to generate mixed prompts
def generate_mixed_prompts(jsonl_stream, num_prompts):
    mixed_prompts = []
    for _ in range(num_prompts):
        try:
            jsonl_content = next(jsonl_stream)["full_clause"]
            preestablished_prompt = random.choice(preestablished_prompts)
            mixed_prompt = preestablished_prompt.format(jsonl_content)
            mixed_prompts.append(mixed_prompt)
        except StopIteration:
            break
    return mixed_prompts


# Generate mixed prompts
jsonl_stream = stream_jsonl("original_dataset.jsonl")
mixed_prompts = generate_mixed_prompts(jsonl_stream, 100)  # Generate 100 mixed prompts

# Create a dataset from mixed prompts
dataset = Dataset.from_dict({"input": mixed_prompts})

# Generate haiku prompts
distiset = await pipeline.generate(
    dataset=dataset,
    num_generations=1,
    shuffle_before_labelling=False,
    batch_size=4,
    display_progress_bar=True,
)


# Clean up the generated prompts
def transform(inst: str) -> str:
    """Remove 1., 2., ... from the instruction."""
    clean_inst = re.sub(r"^\d+\.\s*", "", inst)
    return f"{clean_inst}"


# Process and append results to a new jsonl file
with open("generated_haiku_prompts.jsonl", "w") as outfile:
    for i, generations in enumerate(distiset["raw_generation_responses"]):
        for prompt in generations[0].split("\n"):
            if prompt != "":
                clean_prompt = transform(prompt)
                result = {
                    "original_input": dataset[i]["input"],
                    "generated_prompt": clean_prompt,
                }
                json.dump(result, outfile)
                outfile.write("\n")

print("Generated prompts have been saved to generated_haiku_prompts.jsonl")
