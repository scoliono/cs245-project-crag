import bz2
import json
import os
from datetime import datetime
import argparse

from loguru import logger
from tqdm.auto import tqdm

import numpy as np
import ray
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from collections import defaultdict

import sys
sys.path.append("/content/reasoning-on-cots")

from src.common.config import Config
from src.gpt3_accessors.gpt_accessor_factory import GptAccessorFactory
from src.prompting.prompt_factory import PromptFactoryDict


def load_data_in_batches(dataset_path, batch_size, split=-1):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)

                    if split != -1 and item["split"] != split:
                        continue

                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


# Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
sentence_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
)

def calculate_embeddings(sentences):
    """
    Compute normalized embeddings for a list of sentences using a sentence encoding model.

    This function leverages multiprocessing to encode the sentences, which can enhance the
    processing speed on multi-core machines.

    Args:
        sentences (List[str]): A list of sentences for which embeddings are to be computed.

    Returns:
        np.ndarray: An array of normalized embeddings for the given sentences.

    """
    embeddings = sentence_model.encode(
        sentences=sentences,
        normalize_embeddings=True,
        batch_size=32,
    )
    # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
    #       but sentence_model.encode_multi_process seems to interefere with Ray
    #       on the evaluation servers. 
    #       todo: this can also be done in a Ray native approach.
    #       
    return embeddings


# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids



chunk_extractor = ChunkExtractor()

def generate_predictions(dataset_path, prompt_name, model, temp, split):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    model (object): GptAccessor that provides `call_gpt()` interface.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = 32
    prompt_template = PromptFactoryDict[prompt_name]

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size, split), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        

        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)



        batch_predictions = []
        for idx, query in enumerate(batch["query"]):
            # format prompt with retrieved context
            full_prompt = prompt_template
            if len(batch_retrieval_results[idx]) > 0:
                full_prompt += "\nContext1: " + batch_retrieval_results[idx][0]
            full_prompt += "\nQuestion: " + query
            full_prompt += "\nAre follow up questions needed here: No."
            full_prompt += "\nSo the final answer is: \n"


            batch_predictions.append(
                model.call_gpt(full_prompt, '#', temp)
            )

        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

    return queries, ground_truths, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="example_data/dev_data.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2", # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2", # full data
                                 ])
    parser.add_argument("--split", type=int, default=-1,
                        help="The split of the dataset to use. This is only relevant for the full data: "
                             "0 for public validation set, 1 for public test set")

    parser.add_argument("--accessor_name", type=str, default="gpt_accessor_simple_retrobust",
                        choices=[
                          "gpt_accessor_simple",
                          "gpt_accessor_with_retrieval",
                          "gpt_accessor_with_retrieval_context_first",
                          "gpt_accessor_simple_retrobust",
                          "gpt_accessor_with_retrieval_context_first_retrobust"
                        ]
    )


    parser.add_argument("--temperature", type=float, default=0.3)

    parser.add_argument("--model_name", type=str, default="retrobust_baseline",
                        choices=[# add your model here
                                 "retrobust_baseline"
                                 ],
                        )


    parser.add_argument("--prompt_name", type=str, default="nq_with_retrieval_at1",
                        choices=[
                                 "nq_no_retrieval",
                                 "nq_with_retrieval_at1",
                                 "nq_with_retrieval_at10",
                                 "nq_with_retrieval_mix"
                                ],
                        )

    # Used to determine if we are doing a single-hop vs multi-hop task
    parser.add_argument("--dataset_name", type=str, default="nq",
                        choices=["nq", "strategyqa", "wikihop"]
    )

    # We are using fastchat/openai backends instead of vllm
    parser.add_argument("--llm_wrapper", type=str, default="fastchat",
                        choices=["fastchat", "openai"]
    )


    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-2-13b-hf",
                        choices=[# can add more llm models here
                                 "meta-llama/Llama-2-13b-hf",
                                 ])
    parser.add_argument("--is_server", action="store_true", default=True,
                        help="Whether we use vLLM deployed on a server or offline inference.")

    args = parser.parse_args()
    print(args.is_server)

    Config().load('/content/reasoning-on-cots/src/config/retrobust/nq/with_retrieval_top_1.json')

    Config().override_dict({
        "decomposition.llm_wrapper": args.llm_wrapper,
        "dataset.name": args.dataset_name
    })

    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[0]
    split = -1
    if dataset == "data":
        split = args.split
        if split == -1:
            raise ValueError("Please provide a valid split value for the full data: "
                             "0 for public validation set, 1 for public test set.")
    dataset_path = os.path.join("..", dataset_path)

    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]
    
    model_name = args.model_name
    if model_name == "retrobust_baseline":
        model = GptAccessorFactory().get_instance(args.accessor_name)
    else:
        raise ValueError("Model name not recognized.")

    # make output directory
    output_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
    os.makedirs(output_directory, exist_ok=True)

    # Generate predictions
    queries, ground_truths, predictions = generate_predictions(dataset_path, args.prompt_name, model, args.temperature, split)

    # save predictions
    json.dump({"queries": queries, "ground_truths": ground_truths, "predictions": predictions},
              open(os.path.join(output_directory, "predictions.json"), "w"), indent=4)
