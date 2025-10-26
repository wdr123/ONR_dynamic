#!/usr/bin/env python3
import os
import shutil

input_file_list = [
        ".keepfile",
        "array_input.dat",
        "ccx",
        "cdl.dat",
        "cleanup.py",
        "copy_FSI_inp.sh",
        "FSI_case_001.inp",
        "input_read_flow.dat",
        "input-whisker-signal.dat",
        "materials.inp",
        "README.md",
        "wageng",
        "whisker_001.cab",
        "whisker_001.beam.inp",
        "whisker.cab",
        "whisker_array_driver.py",
        ]

def cleanup(directory,exceptions):
    # todo
    # make a robust script that can do this recursively
    ...
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if item not in exceptions:  # Check against exceptions
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        print(f"Removed file: {item_path}")
                    except OSError as e:
                        print(f"Error removing file {item_path}: {e}")
                elif os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)  # Use shutil.rmtree for directories
                        print(f"Removed directory: {item_path}")
                    except OSError as e:
                        print(f"Error removing directory {item_path}: {e}")


    except OSError as e:
        print(f"An error occurred: {e}")

    return

if __name__ == "__main__":
    cleanup(".",input_file_list)


