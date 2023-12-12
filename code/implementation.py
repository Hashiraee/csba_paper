"""
This file contains the implementation of the Product, FeaturesMap, and ProductData classes for televisions.
These classes are used to represent and manipulate data related to televisions.

The file also contains the implementation of the following functions:
- load_data: Loads the product data from a JSON file.
- normalize_text: Normalizes the input text by removing certain characters and replacing multiple spaces with a single space.
- normalize_units: Normalizes the units in the input text.
- convert_fraction_to_decimal: Converts fractions in the input string to decimal form.
- normalize_dimensions: Normalizes the dimensions or resolution in the input string.
- normalize_brightness: Normalizes the brightness in the input string.
- remove_store: Removes the store names from the input string.
- preprocess_data: Preprocesses the product data.
- process_data: Processes the product data.
- transform: Transforms the input string by converting it to lower case, removing leading and trailing spaces, and replacing all spaces with underscores.
- convert: Converts the product data into a preprocessed format.
- generate_hash_functions: Generates a list of hash functions.
- compute_minhash_signature: Computes the MinHash signature for a set of hashed shingles.
- generate_shingles: Generates k-shingles from the given text.
- hash_shingles: Hashes each shingle into a fixed-size integer using a hash function.
- initialize_lsh: Initializes the Locality Sensitive Hashing (LSH) structure.
- apply_lsh: Applies Locality Sensitive Hashing (LSH) to the given signatures using the provided LSH structure.
- find_candidate_pairs: Finds candidate pairs of products that may be duplicates.
- calculate_similarity: Calculates the Jaccard similarity between two sets.
- classify_duplicates: Classifies candidate pairs as duplicates or non-duplicates based on their Jaccard similarity.
- bootstrap_sampling: Performs bootstrapping on the dataset and separates training and test data.
- get_duplicates: Identifies the actual duplicates in the data based on the productID or modelID.
- evaluate_performance: Evaluates the performance of the duplicate detection algorithm using the F1-score.
- calculate_pair_quality: Calculates the pair quality metric.
- calculate_pair_completeness: Calculates the pair completeness metric.
- calculate_f1_star: Calculates the F1-star metric.
- main: The main function of the program.

Authors:
    - Hasan Israeli (this code, implementation, product representation and analysis)
    - Ziad Massali (shares the base architecture of the code)
"""


import random
import collections
import fractions
import json
import re
from typing import List
from tqdm import tqdm
import numpy as np


class FeaturesMap:
    """
    A class used to represent the features of a television.

    Attributes
    ----------
    **entries : dict
        A dictionary containing feature names as keys and their corresponding values
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Product:
    """
    A class used to represent a (unique) television which can contain duplicates.

    Attributes
    ----------
    shop : str
        The name of the shop where the television is sold
    url : str
        The URL of the television
    modelID : str
        The model ID of the television
    featuresMap : dict
        A dictionary containing feature names as keys and their corresponding values
    title : str
        The title of the television
    """
    def __init__(self, shop: str, url: str, modelID: str, featuresMap: dict, title: str):
        self.shop = shop
        self.url = url
        self.modelID = modelID
        self.featuresMap = FeaturesMap(**featuresMap)
        self.title = title


class ProductData:
    """
    A class used to represent the ProductData.

    Attributes
    ----------
    productID : str
        the product ID
    products : List[Product]
        a list of Product objects
    """
    def __init__(self, productID: str, products: List[Product]):
        self.productID = productID
        self.products = products


# UNIT_MAP is a dictionary that maps various unit representations to a standardized/normalized form
UNIT_MAP = {
    'inches': 'inch', '"': 'inch', '-inch': 'inch', ' inch': 'inch', 'inch': 'inch',
    ' hz': 'hz', '-hz': 'hz', 'hz': 'hz', 'hertz': 'hz', ' hertz': 'hz', 'hertz.': 'hz', ' hertz.': 'hz', 'hz.': 'hz', ' hz.': 'hz', 
    ' lbs': 'lbs', ' lbs.': 'lbs', 'lbs.': 'lbs', 'lb': 'lbs', 'lb.': 'lbs',
    'pounds': 'lbs', ' pounds': 'lbs', 'pound' : 'lbs',
    'wi-fi': 'wifi', 'wi fi': 'wifi',
    'diag.': 'diagonal', ' diag.' : 'diagonal', ' diag': 'diagonal', 'diag': 'diagonal',
}

# STORE_MAP is a list of stores used for cleaning the titles
STORE_MAP = ["bestbuy.com", "best-buy", "best buy", "newegg", "newegg.com" "amazon", "amazon.com", "thenerds.net", "the nerds"]


def load_data(file_path: str) -> List[ProductData]:
    """
    This function loads the product data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file containing the product data.

    Returns:
    List[ProductData]: A list of ProductData objects, each representing a unique product.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    product_data_list = []
    for productID, products in data.items():
        product_objects = [Product(**product) for product in products]
        product_data = ProductData(productID=productID, products=product_objects)
        product_data_list.append(product_data)
    return product_data_list


def normalize_text(text: str) -> str:
    """
    This function normalizes the input text by removing certain characters and replacing multiple spaces with a single space.

    Parameters:
    text (str): The input string to be normalized.

    Returns:
    str: The normalized string.
    """
    text = text.replace(',', '')
    text = re.sub(r'\(.*?\)', '', text)
    text = text.replace(';', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('-', '')
    text = re.sub(r'[^a-zA-Z0-9+$.:\s/_\\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_units(text: str, unit_map: dict[str, str]) -> str:
    """
    This function normalizes the units in the input text. It splits the text into words, 
    replaces any occurrence of a digit followed by a double quote (") with the digit followed by 'inch', 
    and then replaces each word with its corresponding value in the unit_map.

    Parameters:
    text (str): The input string in which units are to be normalized.
    unit_map (dict[str, str]): A dictionary mapping units to their normalized form.

    Returns:
    str: The input string with all units normalized/standardized.
    """
    words = text.split()
    normalized_words = []
    for word in words:
        word = re.sub(r'(\d+)"', r'\1inch', word)
        normalized_words.append(unit_map.get(word, word))
    return ' '.join(normalized_words)


def convert_fraction_to_decimal(value: str) -> str:
    """
    This function converts fractions in the input string to decimal form.

    Parameters:
    value (str): The input string in which fractions are to be converted to decimal.

    Returns:
    str: The input string with all fractions converted to decimal.
    """
    matches = re.findall(r'(\d+)-(\d+)/(\d+)inch', value)
    for match in matches:
        whole, numerator, denominator = map(int, match)
        decimal = whole + float(fractions.Fraction(numerator, denominator))
        value = value.replace(f'{whole}-{numerator}/{denominator}inch', f'{decimal:.1f}inch')
    return value


def normalize_dimensions(value: str) -> str:
    """
    This function normalizes the dimensions or resolution in the input string. It replaces any occurrence of a dimension 
    (expressed as 'resolution x resolution' or 'numbermm x numbermm') with 'resolution_x_resolution' or 'numbermm_x_numbermm'.

    Parameters:
    value (str): The input string in which dimensions are to be normalized.

    Returns:
    str: The input string with all dimensions normalized.
    """
    value = re.sub(r'(\d+mm|\d+)\s*x\s*(\d+mm|\d+)', r'\1_x_\2', value)
    return value


def normalize_brightness(value: str) -> str:
    """
    This function normalizes the brightness in the input string. It replaces any occurrence of a brightness value 
    (expressed as a number followed by any characters) with the brightness value followed by 'nits'.

    Parameters:
    value (str): The input string in which brightness is to be normalized.

    Returns:
    str: The input string with brightness normalized.
    """
    value = re.sub(r'(\d+).*', r'\1nits', value)
    return value


def remove_store(value: str, store_map: list[str]) -> str:
    """
    This function removes the store names from the input string. It iterates over each store in the store_map list, 
    and replaces any occurrence of the store name in the input string with an empty string.

    Parameters:
    value (str): The input string from which store names are to be removed.
    store_map (list[str]): A list of store names.

    Returns:
    str: The input string with all store names removed.
    """
    for store in store_map:
        value = value.replace(store, '')
    return value


def preprocess_data(data: List[ProductData]) -> List:
    """
    This function preprocesses the product data. It iterates over each product in the data list, 
    extracts the features, and appends them to a new list.

    Parameters:
    data (List[ProductData]): A list containing ProductData objects, each representing a unique product

    Returns:
    List: A list of dictionaries, where each dictionary represents the preprocessed features of a product
    """
    pre_processed_data = []
    for product_data in data:
        for product in product_data.products:
            features = product.featuresMap.__dict__
            features['shop'] = product.shop
            features['url'] = product.url
            features['modelID'] = product.modelID
            features['title'] = product.title
            pre_processed_data.append(features)
    return pre_processed_data


def process_data(data: List) -> List:
    """
    This function processes the product data. It iterates over each product in the data list, 
    normalizes the units, brightness, dimensions, and text, and appends them to a new list.

    Parameters:
    data (List): A list of dictionaries, where each dictionary represents the preprocessed features of a product.

    Returns:
    List: A list of dictionaries, where each dictionary represents the processed features of a product.
    """
    processed_data = []
    for product in data:
        for key, value in product.items():
            if key != 'modelID':
                value = value.lower()
                value = normalize_units(value, UNIT_MAP)
            if key == 'Brightness':
                value = normalize_brightness(value)
            if key == 'title':
                value = remove_store(value, STORE_MAP)
            if 'x' in value:
                value = normalize_dimensions(value)
            for unit in UNIT_MAP.keys():
                value = re.sub(r'(\d+)\s+' + re.escape(unit), r'\1' + unit, value)
                value = convert_fraction_to_decimal(value)
            product[key] = transform(normalize_text(value))
        processed_data.append(product)
    return processed_data


def transform(value: str) -> str:
    """
    This function transforms the input string by converting it to lower case, removing leading and trailing spaces, 
    and replacing all spaces with underscores.

    Parameters:
    value (str): The input string to be transformed.

    Returns:
    str: The transformed string.
    """
    text = value.lower().strip().replace(' ', '_')
    return text


def convert(data: List) -> List:
    """
    This function converts the product data into a preprocessed format. It combines the television product details into a single string,
    tokenizes the text, and adds the preprocessed text to a list.

    Parameters:
    data (List): A list of dictionaries, where each dictionary represents the features of a product.

    Returns:
    List: A list of dictionaries, where each dictionary contains the productID as the key and the preprocessed text as the value.
    """
    preprocessed_data = []
    for product in data:
        # Combine product details into a single string
        product_items = []
        for k, v in product.items():
            if k not in ['shop', 'url', 'modelID']:
                key = transform(normalize_text(k)).lower()
                value = v
                product_items.append(f'{key}_{value}')
        combined_text = ' '.join(product_items)

        # Tokenize the text
        tokenized_text = re.findall(r'\b[\w:.]+\b', combined_text)
        # Add the preprocessed text to the list along with the productID
        preprocessed_data.append({product['modelID']: ' '.join(tokenized_text)})
    return preprocessed_data


def generate_hash_functions(num_hash_functions: int) -> list:
    """
    This function generates a list of hash functions.

    Parameters:
    num_hash_functions (int): The number of hash functions to generate.

    Returns:
    list: A list of generated hash functions.
    """
    def generate_hash_function():
        """
        This nested function generates a single hash function.

        Returns:
        function: A hash function.
        """
        a = random.randint(1, 2**32 - 1)
        b = random.randint(0, 2**32 - 1)
        return lambda x: (a * x + b) % (2**32)

    return [generate_hash_function() for _ in range(num_hash_functions)]


def compute_minhash_signature(hashed_shingles: set, hash_functions: list) -> list:
    """
    This function computes the MinHash signature for a set of hashed shingles.

    The MinHash signature is a list where each element is the minimum hash value 
    obtained by applying a hash function to the set of hashed shingles.

    Parameters:
    hashed_shingles (set): The set of hashed shingles.
    hash_functions (list): A list of hash functions.

    Returns:
    list: The MinHash signature for the set of hashed shingles.
    """
    return [min(hash_function(shingle) for shingle in hashed_shingles) for hash_function in hash_functions]


def generate_shingles(data: str, k: int) -> set:
    """
    Generate k-shingles from the given text.
    A k-shingle is a substring of 'k' consecutive characters from the text.
    
    Parameters:
    data (str): The text from which to generate k-shingles.
    k (int): The length of each shingle.
    
    Returns:
    shingles (set): The set of k-shingles generated from the text.
    """
    shingles = set()
    for i in range(len(data) - k + 1):
        shingle = data[i:i + k]
        shingles.add(shingle)
    return shingles


def hash_shingles(shingles: set) -> set:
    """
    Hashes each shingle into a fixed-size integer using a hash function.
    The built-in hash function is used for efficiency.
    
    Parameters:
    shingles (set): The set of k-shingles generated from the text.
    
    Returns:
    hashed_shingles (set): The set of hashed shingles.
    """
    hashed_shingles = set()
    for shingle in shingles:
        hashed_shingle = hash(shingle) % (2**32)
        hashed_shingles.add(hashed_shingle)
    return hashed_shingles


def initialize_lsh(num_bands: int, rows_per_band: int) -> dict:
    """
    Initializes the Locality Sensitive Hashing (LSH) structure.

    The LSH structure is a dictionary that contains the number of bands, the number of rows per band, 
    and a list of empty dictionaries (buckets) for each band.

    Parameters:
    num_bands (int): The number of bands for the LSH structure.
    rows_per_band (int): The number of rows per band for the LSH structure.

    Returns:
    dict: The initialized LSH structure.
    """
    lsh_structure = {
        'num_bands': num_bands,
        'rows_per_band': rows_per_band,
        'buckets': [{} for _ in range(num_bands)]
    }
    return lsh_structure


def apply_lsh(signatures, lsh_structure: dict) -> dict:
    """
    Applies Locality Sensitive Hashing (LSH) to the given signatures using the provided LSH structure.

    The function divides each signature into bands and hashes each band into a bucket. 

    Parameters:
    signatures: A list of dictionaries, where each dictionary represents a
    product with the productID as the key and the MinHash signature as the value.

    lsh_structure (dict): The LSH structure to be used, which includes the
    number of bands, the number of rows per band, and the buckets.

    Returns:
    dict: The updated LSH structure
    """
    num_bands = lsh_structure['num_bands']
    rows_per_band = lsh_structure['rows_per_band']
    buckets = lsh_structure['buckets']

    for product in signatures:
        for model_id, signature in product.items():
            for band in range(num_bands):
                # Extract the portion of the signature for this band
                start_index = band * rows_per_band
                end_index = start_index + rows_per_band
                band_signature = tuple(signature[start_index:end_index])
                
                # Hash the band signature to a bucket
                bucket_hash = hash(band_signature) % (2**32)
                
                # Add the modelID to the corresponding bucket
                if bucket_hash not in buckets[band]:
                    buckets[band][bucket_hash] = []
                buckets[band][bucket_hash].append(model_id)

    lsh_structure['buckets'] = buckets
    return lsh_structure


def find_candidate_pairs(lsh_structure: dict) -> set:
    """
    Finds candidate pairs of products that may be duplicates.

    This function iterates over each band in the LSH structure, and for each band, it iterates over each bucket.
    If a bucket contains more than one product, it is possible that these products are duplicates.
    The function generates all possible pairs of products within each such bucket and adds them to a set of candidate pairs.

    Parameters:
    lsh_structure (dict): The LSH structure to be used, which includes the
    number of bands, the number of rows per band, and the buckets.

    Returns:
    set: A set of candidate pairs of products that may be duplicates.
    """
    candidate_pairs = set()
    for band_buckets in lsh_structure['buckets']:
        for bucket in band_buckets.values():
            if len(bucket) > 1:
                # Sorting the bucket to ensure consistent order of pairs
                sorted_bucket = sorted(bucket)
                for i in range(len(sorted_bucket)):
                    for j in range(i + 1, len(sorted_bucket)):
                        # Adding the sorted pair to the set
                        candidate_pairs.add((sorted_bucket[i], sorted_bucket[j]))
                        
    return candidate_pairs


def calculate_similarity(set1: set, set2: set) -> float:
    """
    Calculates the Jaccard similarity between two sets.

    The Jaccard similarity is defined as the size of the intersection divided by the size of the union of the two sets.

    Parameters:
    set1 (set): The first set.
    set2 (set): The second set.

    Returns:
    float: The Jaccard similarity between the two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def classify_duplicates(candidates: set, signatures: dict, threshold: float) -> set:
    """
    Classifies candidate pairs as duplicates or non-duplicates based on their Jaccard similarity.

    This function iterates over each candidate pair and retrieves their corresponding MinHash signatures.
    It then calculates the Jaccard similarity between the two signatures.
    If the similarity is greater than a specified threshold, the pair is classified as a duplicate.

    Parameters:
    candidates (set): A set of candidate pairs of products that may be duplicates.
    signatures (dict): A dictionary where each key is a product and the value is its MinHash signature.
    threshold (float): The similarity threshold for classifying a pair as a duplicate.

    Returns:
    set: A set of pairs that are classified as duplicates.
    """
    duplicates = set()
    for candidate in candidates:
        set1 = None
        set2 = None
        for product in signatures:
            if candidate[0] in product:
                set1 = set(product[candidate[0]])
            if candidate[1] in product:
                set2 = set(product[candidate[1]])
            if set1 and set2:
                break
        if set1 and set2:
            similarity = calculate_similarity(set1, set2)
            if similarity > threshold:
                duplicates.add(candidate)
    return duplicates


def bootstrap_sampling(data: list, num_bootstraps: list) -> list:
    """
    Performs bootstrapping on the dataset and separates training and test data.
    
    Parameters:
    data (list): The dataset.
    num_bootstraps (int): The number of bootstrap samples to generate.
    
    Returns:
    bootstrap_samples (list): A list of tuples, where each tuple contains a training set and a test set.
    """
    bootstrap_samples = []
    train_size = len(data)
    
    for _ in range(num_bootstraps):
        train_data = np.random.choice(data, train_size, replace=True)
        test_data = [instance for instance in data if instance not in train_data]
        bootstrap_samples.append((train_data, test_data))
    
    return bootstrap_samples


def get_duplicates(data: list) -> dict:
    """
    Identifies the actual duplicates in the data based on the productID or modelID.
    
    Parameters:
    data (list): The dataset.
    
    Returns:
    duplicates (dict): A dictionary where the keys are modelIDs and the values are lists of all products with that productID or modelID.
    """
    duplicates = collections.defaultdict(list)
    for product in data:
        for product_id, _ in product.items():
            duplicates[product_id].append(product)
    return duplicates


def evaluate_performance(predictions: list, ground_truth: list) -> float:
    """
    Evaluates the performance of the duplicate detection algorithm using the F1-score.
    
    Parameters:
    predictions (list): The list of predicted duplicate pairs.
    ground_truth (list): The list of actual duplicate pairs.
    data (dict): The original data loaded from the JSON file. Each key is a model_id and the value is a list of products.
    
    Returns:
    f1_score (float): The F1-score of the predictions.
    """
    TP = len([pair for pair in predictions if pair in ground_truth])
    FP = len([pair for pair in predictions if pair not in ground_truth])
    FN = len([pair for pair in ground_truth if pair not in predictions])
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    
    # Calculating the F1-score as defined in the assignment
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1_score


def calculate_pair_quality(duplicates: set, comparisons: int) -> float:
    """
    Calculates the pair quality metric.
    
    The pair quality is defined as the number of true duplicate pairs divided by the total number of comparisons made.
    
    Parameters:
    duplicates (set): The set of true duplicate pairs identified.
    comparisons (int): The total number of comparisons made.
    
    Returns:
    pair_quality (float): The pair quality metric.
    """
    pair_quality = len(duplicates) / comparisons if comparisons > 0 else 0
    return pair_quality


def calculate_pair_completeness(duplicates: set, total_duplicates: int) -> float:
    """
    Calculates the pair completeness metric.
    
    The pair completeness is defined as the number of true duplicate pairs identified divided by the total number of duplicate pairs.
    
    Parameters:
    duplicates (set): The set of true duplicate pairs identified.
    total_duplicates (int): The total number of duplicate pairs.
    
    Returns:
    pair_completeness (float): The pair completeness metric.
    """
    pair_completeness = len(duplicates) / total_duplicates if total_duplicates > 0 else 0
    return pair_completeness


def calculate_f1_star(pair_quality: float, pair_completeness: float) -> float:
    """
    Calculates the F1-star metric.
    
    The F1-star metric is defined as the harmonic mean of pair quality and pair completeness.
    If the sum of pair quality and pair completeness is zero, the F1-star metric is defined to be zero.
    
    Parameters:
    pair_quality (float): The pair quality metric.
    pair_completeness (float): The pair completeness metric.
    
    Returns:
    f1_star (float): The F1-star metric.
    """
    if pair_quality + pair_completeness > 0:
        f1_star = 2 * (pair_quality * pair_completeness) / (pair_quality + pair_completeness)
    else:
        f1_star = 0
    return f1_star


def main():
    # Setting seed for reproduction
    np.random.seed(42)

    # Loading the data
    file_path = 'data/data.json'
    product_data_list = load_data(file_path)
    
    # Processsing the data
    preprocessed_data = preprocess_data(product_data_list)
    processed_data = process_data(preprocessed_data)
    data = convert(processed_data)

    # Setting parameters for shingles, LSH, and similarity threshold
    k_shingles = 10
    num_hash_functions = 100
    # num_bands = 5
    # rows_per_band = 20
    num_bands = 4
    rows_per_band = 25
    similarity_threshold = (1 / num_bands) ** (1 / rows_per_band)
    print(f"Similarity threshold: {similarity_threshold}")

    # Generating hash functions for MinHash signatures
    hash_functions = generate_hash_functions(num_hash_functions)

    # Bootstrap sampling and evaluation
    num_bootstraps = 5
    f1_scores = []
    pair_qualities = []
    pair_completenesses = []
    f1_stars = []

    bootstrap_samples = bootstrap_sampling(data, num_bootstraps)
    for i, (train_data, test_data) in enumerate(bootstrap_samples):
        print(f"Processing bootstrap sample {i + 1}...")

        # Generating shingles and hashing them for each product in the training set
        print("Generating shingles and hashing them for each product in the training set...")
        train_signatures = []
        for product in tqdm(train_data, desc="Generating shingles for training"):
            for model_id, text in product.items():
                # Generating shingles based on our representation
                shingles = generate_shingles(text, k_shingles)
                
                # Hashing the shingles
                hashed_shingles = hash_shingles(shingles)

                # Computing MinHash signatures
                minhash_signature = compute_minhash_signature(hashed_shingles, hash_functions)
                train_signatures.append({model_id: minhash_signature})

        # Applying LSH to the training signatures
        print("Applying LSH to the training signatures...")
        train_lsh_structure = initialize_lsh(num_bands, rows_per_band)
        train_lsh_structure = apply_lsh(train_signatures, train_lsh_structure)

        # Generating shingles and hashing them for each product in the test set
        print("Generating shingles and hashing them for each product in the test set...")
        test_signatures = []
        for product in tqdm(test_data, desc="Generating shingles for testing"):
            for model_id, text in product.items():
                # Generating shingles based on our representation
                shingles = generate_shingles(text, k_shingles)
                
                # Hashing the shingles
                hashed_shingles = hash_shingles(shingles)

                # Computing MinHash signatures
                minhash_signature = compute_minhash_signature(hashed_shingles, hash_functions)
                test_signatures.append({model_id: minhash_signature})


        # Finding the candidate pairs using LSH structure from the training data
        print("Applying LSH to the testing signatures...")
        test_candidate_pairs = find_candidate_pairs(train_lsh_structure)
        print(f"Candidate duplicates in test data: {len(test_candidate_pairs)}")

        # Classifying duplicates from candidate pairs in the test data
        print("Classifying duplicates from candidate pairs in the test data...")
        test_duplicates = classify_duplicates(test_candidate_pairs, test_signatures, similarity_threshold)
        print(f"Classified duplicates in test data: {len(test_duplicates)}")

        # Getting the ground truth for the test set
        print("Getting ground truth for the test set...")
        ground_truth_test = get_duplicates(test_data)
        ground_truth_test_tuples = [(product, product) for product in ground_truth_test]
        print(f"Number of actual duplicates: {len(ground_truth_test_tuples)}")

        # Evaluating the performance on the test set
        print("Evaluating performance on the test set...")
        f1_score = evaluate_performance(test_duplicates, ground_truth_test_tuples)
        f1_scores.append(f1_score)

        # Calculating the pair quality and pair completeness
        print("Calculating pair quality and completeness...")
        pair_quality = calculate_pair_quality(test_duplicates, len(test_candidate_pairs))
        pair_completeness = calculate_pair_completeness(test_duplicates, len(ground_truth_test_tuples))
        pair_qualities.append(pair_quality)
        pair_completenesses.append(pair_completeness)

        # Calculating the F1* measure
        print("Calculating F1* measure...")
        f1_star = calculate_f1_star(pair_quality, pair_completeness)
        f1_stars.append(f1_star)

    
    # Calculating the average metrics across bootstraps
    print("Calculating average metrics across bootstraps...")
    avg_f1_score = np.mean(f1_scores)
    avg_pair_quality = np.mean(pair_qualities)
    avg_pair_completeness = np.mean(pair_completenesses)
    avg_f1_star = np.mean(f1_stars)
    
    # Outputting the evaluation metrics
    print("Outputting the evaluation results...")
    print(f"Average F1-score: {avg_f1_score}")
    print(f"Average Pair Quality: {avg_pair_quality}")
    print(f"Average Pair Completeness: {avg_pair_completeness}")
    print(f"Average F1* measure: {avg_f1_star}")


if __name__ == '__main__':
    main()
