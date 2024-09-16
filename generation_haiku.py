# generation_haiku.py

import asyncio
import json
import os
import random
import re
import logging
from typing import List

import openai
from dotenv import load_dotenv

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import TextGeneration

# Load environment variables
load_dotenv()

# Configure simple logging to a file
logging.basicConfig(
    filename="generatlogg2.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# Function to stream content from original_dataset.jsonl
def stream_jsonl(file_path: str):
    logger.debug(f"Streaming JSONL from {file_path}")
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

# Create the LLM (using OpenAI API compatible endpoint)
llm = OpenAILLM(
    model=os.getenv("model"),
    base_url=os.getenv("base_url"),
    api_key=os.getenv("api_key"),
)


# Function to generate mixed prompts
def generate_mixed_prompts(jsonl_stream, num_prompts: int) -> List[str]:
    mixed_prompts = []
    for _ in range(num_prompts):
        try:
            jsonl_content = next(jsonl_stream)["full_clause"]
            preestablished_prompt = random.choice(preestablished_prompts)
            mixed_prompt = f"You are an expert Haiku writer. {preestablished_prompt.format(jsonl_content)}"
            mixed_prompts.append(mixed_prompt)
        except StopIteration:
            logger.debug("Reached the end of JSONL stream.")
            break
    logger.debug(f"Generated {len(mixed_prompts)} mixed prompts.")
    return mixed_prompts


# Clean up the generated prompts
def transform(inst: str) -> str:
    clean_inst = re.sub(r"^\d+\.\s*", "", inst)
    return f"{clean_inst}"


async def main():
    logger.info("Starting main function.")
    # Generate mixed prompts
    jsonl_stream = stream_jsonl("original_dataset.jsonl")
    mixed_prompts = generate_mixed_prompts(
        jsonl_stream, 100
    )  # Generate 100 mixed prompts

    # Log generated prompts
    logger.debug(f"Generated Prompts: {mixed_prompts}")

    # Create the pipeline
    with Pipeline(name="text_generation") as pipeline:
        logger.debug("Pipeline created.")

        load_data = LoadDataFromDicts(
            name="load_data",
            data=[{"input": prompt} for prompt in mixed_prompts],
            output_mappings={"input": "instruction"},
        )
        logger.debug(
            f"Data for LoadDataFromDicts: {[{'input': prompt} for prompt in mixed_prompts]}"
        )

        text_generation = TextGeneration(
            name="text_generation",
            llm=llm,
            use_system_prompt=False,
            input_batch_size=10,
            output_mappings={"model_name": "generation_model"},
        )
        logger.debug("Configured TextGeneration step.")

        keep_columns = KeepColumns(
            name="keep_columns",
            columns=[
                "instruction",
                "generation",
            ],
        )
        logger.debug("Configured KeepColumns step.")

        load_data >> text_generation >> keep_columns

        # Run the pipeline and return the result
        result = pipeline.run()
        logger.info("Pipeline run completed.")

        # Stream the result to a JSONL file
        with open("generated_haiku_prompts.jsonl", "a") as outfile:
            for item in result["default"]["train"]:
                json.dump(item, outfile)
                outfile.write("\n")
        logger.info("Generated haiku prompts saved to generated_haiku_prompts.jsonl")

        return result


if __name__ == "__main__":
    logger.info("Script execution started.")
    result = asyncio.run(main())

    if result:
        logger.debug("Processing and saving results.")
        with open("generated_haiku_prompts.jsonl", "a") as outfile:
            for item in result["default"]["train"]:
                json.dump(item, outfile)
                outfile.write("\n")
        logger.info("Generated haiku prompts saved to generated_haiku_prompts.jsonl")
    else:
        logger.warning("No data generated.")

    # Properly shutdown the logging system
    logging.shutdown()
