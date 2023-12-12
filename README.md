# Computer Science for Business Analytics Assignment

## Title
LSH-Based Shingles-Similarity Product Duplication Detection Method

## Overview of the Project
```
.
├── Makefile                # Makefile for easy setup and installation
├── README.md               # This README file
├── code
│   └── implementation.py   # Implementation of the solution
├── data
│   └── data.json           # The television data for the project
├── report                      
│   ├── report.pdf          # The actual report itself
│   ├── report.tex          # The Latex Code for the report
│   └── ...
└── requirements.txt        # The requirements for the code

```

## Installation
#### Requirements (Python code)
- Python 3 (pyton3)
- pip (pip3)

#### Installation Steps
1. Clone the repository: `git clone https://github.com/Hashiraee/cs_for_ba_project.git`
2. Setup virtual environment & Install dependencies:
    - `cd cs_for_ba_project`
    - `make install`
    - (Or `pip3 install -r requirements.txt`)

## Usage
To run the code:
- Make sure you are in the root of the `cs_for_ba_project` directory
- `make run`
- Or `python3 code/implementation.py`

## Code structure
**The solution/code is in code/implementation.py and has the following (code) structure:**
1. The implementation of the Product, FeaturesMap, and ProductData classes for
   televisions. These classes are used to represent and manipulate data related
   to televisions.

2. The code file also contains the implementation of the following functions:
    - **load_data:** Loads the product data from a JSON (in data/data.json) file.
    - **normalize_text:** Normalizes the input text by removing certain characters and replacing multiple spaces with a single space.
    - **normalize_units:** Normalizes the units in the input text.
    - **convert_fraction_to_decimal:** Converts fractions in the input string to decimal form.
    - **normalize_dimensions:** Normalizes the dimensions or resolution in the input string.
    - **normalize_brightness:** Normalizes the brightness in the input string.
    - **remove_store:** Removes the store names from the input string.
    - **preprocess_data:** Preprocesses the product data.
    - **process_data:** Processes the product data.
    - **transform:** Transforms the input string by converting it to lower case, removing leading and trailing spaces, and replacing all spaces with underscores.
    - **convert:** Converts the product data into a preprocessed format.
    - **generate_hash_functions:** Generates a list of hash functions.
    - **compute_minhash_signature:** Computes the MinHash signature for a set of hashed shingles.
    - **generate_shingles:** Generates k-shingles from the given text.
    - **hash_shingles:** Hashes each shingle into a fixed-size integer using a hash function.
    - **initialize_lsh:** Initializes the Locality Sensitive Hashing (LSH) structure.
    - **apply_lsh:** Applies Locality Sensitive Hashing (LSH) to the given signatures using the provided LSH structure.
    - **find_candidate_pairs:** Finds candidate pairs of products that may be duplicates.
    - **calculate_similarity:** Calculates the Jaccard similarity between two sets.
    - **classify_duplicates:** Classifies candidate pairs as duplicates or non-duplicates based on their Jaccard similarity.
    - **bootstrap_sampling:** Performs bootstrapping on the dataset and separates training and test data.
    - **get_duplicates:** Identifies the actual duplicates in the data based on the productID or modelID.
    - **evaluate_performance:** Evaluates the performance of the duplicate detection algorithm using the F1-score.
    - **calculate_pair_quality:** Calculates the pair quality metric.
    - **calculate_pair_completeness:** Calculates the pair completeness metric.
    - **calculate_f1_star:** Calculates the F1-star metric.
    - **main:** The main function of the program.
