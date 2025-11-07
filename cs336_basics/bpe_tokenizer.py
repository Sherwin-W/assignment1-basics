import os
import re
import pretokenization_example

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], test_chars: int):
    """
    1. Read corpus from input_path (UTF-8 text file).
    2. Split on special tokens (so they remain intact).
    3. For each piece, pre-tokenize with GPT-2 regex pattern.
    4. Count how often each byte sequence (pre-token) appears.
    5. Count adjacent byte-pair frequencies inside those pre-tokens.
    6. While len(vocab) < vocab_size:
           find most-common pair (break ties lexicographically)
           merge pair into new symbol
           update pair counts efficiently
    7. Return (vocab, merges).
    """
    
    print(f"\n--- Starting BPE training ---")
    print(f"  Input path: {input_path}")
    print(f"  Target vocab size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    
    
    with open(os.path, "rb") as f:
        boundaries = pretokenization_example.find_chunk_boundaries(f, desired_num_chunks=4, split_special_token=b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            print(chunk_bytes)

def main():
    """
    This is the main entry point for testing the script.
    """
    print("--- Running BPE Tokenizer Script ---")

    print("\n--- Testing file read access ---")
    file_path = '/Volumes/Ridge1TB/CS336 Stanford ML/data/TinyStoriesV2-GPT4-train.txt'
    
    dummy_vocab_size = 500
    dummy_special_tokens = ["<|endoftext|>"]
    
    train_bpe(input_path=file_path, vocab_size=dummy_vocab_size, special_tokens=dummy_special_tokens, test_chars=10000)


if __name__ == '__main__':
    main()