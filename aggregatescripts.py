import os

# Define the list of file paths
files = [
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/docs/sections/getting_started/quickstart.md",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/docs/sections/how_to_guides/basic/pipeline/index.md",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/pipeline/base.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/steps/tasks/base.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/steps/decorator.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/steps/generators/data.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/steps/tasks/self_instruct.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/steps/tasks/quality_scorer.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/src/distilabel/steps/base.py",
    "/Users/arthrod/Library/CloudStorage/GoogleDrive-arthursrodrigues@gmail.com/My Drive/aCode/gits/distilabel/examples/structured_generation_with_outlines.py",
]

# Name of the output file
output_filename = "input.txt"

try:
    with open(output_filename, "w", encoding="utf-8") as outfile:
        for file_path in files:
            try:
                # Extract the filename from the path
                filename = os.path.basename(file_path)

                # Write the filename as the first line
                outfile.write(f"{filename}\n")

                # Read the content of the current file
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)

                # Add a newline to separate contents of different files
                outfile.write("\n\n")

                print(f"Successfully added {filename} to {output_filename}")

            except FileNotFoundError:
                print(f"Error: The file '{file_path}' was not found.")
            except Exception as e:
                print(f"An error occurred while processing '{file_path}': {e}")

    print(f"\nAll contents have been written to '{output_filename}'.")

except Exception as e:
    print(f"Failed to write to '{output_filename}': {e}")
