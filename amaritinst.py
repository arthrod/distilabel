from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import SelfInstruct
from typing import Dict, Any, List, Union
import json
import uuid
from typing_extensions import override
from distilabel.steps.tasks.base import Task


# Variables for easy modification
DATASET = "abcd.jsonl"  # Input file containing the clauses
OUTPUT_FILE = "inst_abcd.jsonl"  # Output file for the generated instructions
BATCH_SIZE = 50  # Batch size for processing
MODEL_NAME = "gpt-3.5-turbo"  # OpenAI model to use
MARITACA_API_KEY = "your_api_key"  # Replace with your actual API key
MARITACA_BASE_URL = "your_base_url"  # Replace with your actual base URL if necessary
NUM_INSTRUCTIONS = 5  # Number of instructions to generate per clause


class ProcessSelfInstructOutputs(Task):
    """Custom task to process SelfInstruct outputs."""

    @property
    def outputs(self) -> List[str]:
        return ["_id", "new_id", "clausula", "clause_type", "instructions", "model"]

    def format_input(self, input: Dict[str, Any]) -> Any:
        """Not used since we're processing existing data."""
        pass

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "_id": input["_id"],
            "new_id": str(uuid.uuid4()),
            "clausula": input["clausula"],
            "clause_type": input.get("clause_type"),
            "instructions": output,
            "model": input.get("model_name", self.llm.model_name),
        }

    @override
    def process(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the SelfInstruct outputs."""
        task_outputs = []
        for input_row in inputs:
            try:
                # Assuming the self_instruct task outputs a list of instructions under 'instructions'
                instructions = input_row.get("instructions", [])
                for instruction in instructions[:NUM_INSTRUCTIONS]:
                    formatted_output = self.format_output(instruction, input_row)
                    task_outputs.append(formatted_output)
            except Exception as e:
                self._logger.warning(
                    f"Failed to process input row with _id {input_row.get('_id')}: {e}"
                )
        return task_outputs


if __name__ == "__main__":
    # Load the input data
    try:
        with open(DATASET, "r", encoding="utf-8") as f:
            inputs = [
                {
                    "_id": input_data["_id"],
                    "clausula": input_data["clausula"],
                    "clause_type": input_data.get("clause_type", None),
                    # Optionally include 'model_name' if it's part of your input data
                    "model_name": input_data.get("model_name", MODEL_NAME),
                }
                for input_data in (json.loads(line) for line in f)
            ]
    except FileNotFoundError:
        print(f"Error: The file '{DATASET}' was not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file '{DATASET}': {e}")
        exit(1)

    # Prepare data for the pipeline
    data = inputs  # Use the selected fields from the input data

    # Define the pipeline
    with Pipeline(
        name="rated-self-instructions-for-legal-clauses",
        description="A pipeline to generate instructions based on legal clauses",
        cache_dir=".",  # Use the same directory as the script
        enable_metadata=True,
    ) as pipeline:

        # Step 1: Load data
        load_data = LoadDataFromDicts(
            name="load_data",
            data=data,
            batch_size=BATCH_SIZE,
        )

        # Define the LLM to use
        llm = OpenAILLM(
            model=MODEL_NAME,
            api_key=MARITACA_API_KEY,
            base_url=MARITACA_BASE_URL,
        )

        # Step 2: Generate instructions using SelfInstruct
        self_instruct = SelfInstruct(
            name="self_instruct_generation",
            llm=llm,
            num_instructions=NUM_INSTRUCTIONS,
            application_description=(
                "Você é um profissional jurídico altamente experiente especializado em direito do Brasil. "
                "Sua tarefa é gerar instruções legais precisas, profissionais e autoritativas com base nas cláusulas legais fornecidas."
            ),
            criteria_for_query_generation=(
                "Gere uma instrução que reflita com precisão a essência da cláusula legal. "
                "Certifique-se de que a instrução seja escrita em linguagem jurídica formal, conforme utilizada por profissionais jurídicos no Brasil. "
                "Use terminologia clara e concisa apropriada para documentação legal."
            ),
            input_mappings={
                "clausula": "input"
            },  # Map 'clausula' to 'input' for SelfInstruct
        )

        # Step 3: Custom Task to process SelfInstruct outputs
        process_outputs = ProcessSelfInstructOutputs(
            name="process_self_instruct_outputs",
            llm=llm,  # Ensure that the LLM is passed if used in the Task
        )

        # Step 4: Keep specific columns
        keep_columns = KeepColumns(
            name="keep_columns",
            columns=[
                "_id",
                "new_id",
                "clausula",
                "clause_type",
                "instruction",
                "model",
            ],
        )

        # Connect the steps
        load_data >> self_instruct >> process_outputs >> keep_columns

    # Run the pipeline
    try:
        distiset = pipeline.run()
    except Exception as e:
        print(f"Error running the pipeline: {e}")
        exit(1)

    # Save the pipeline outputs to disk
    try:
        distiset.save_to_disk(
            distiset_path=".",  # Save in the current directory
            save_card=False,  # Set to False if you don't need a card
            save_pipeline_config=True,
            save_pipeline_log=True,
        )
    except Exception as e:
        print(f"Error saving the pipeline outputs: {e}")
        exit(1)

    # Save the pipeline configuration in YAML format
    try:
        pipeline.save("pipeline.yaml", format="yaml")
        print("Pipeline configuration saved to 'pipeline.yaml'")
    except Exception as e:
        print(f"Error saving the pipeline configuration: {e}")

    # Save all of distiset to a JSONL file
    try:
        dataset = distiset["rated-self-instructions-for-legal-clauses"][
            "process_self_instruct_outputs"
        ]
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for item in dataset:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        print(f"Pipeline outputs saved to '{OUTPUT_FILE}'")
    except KeyError as e:
        print(f"Error accessing dataset in distiset: {e}")
    except Exception as e:
        print(f"Error saving the output file: {e}")
