import os
import csv
import json
import re
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from natasha import Segmenter, MorphVocab, Doc, NewsEmbedding, NewsMorphTagger
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Any


class AnglicismReplacer:
    """Class for replacing anglicisms in Russian text with proper Russian synonyms."""

    def __init__(self, anglicisms_file=None, anglicism_dict_file=None,
                 model_name="Qwen/Qwen2.5-3B-Instruct",
                 semantic_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device=None):
        """Initialize the AnglicismReplacer with necessary models and data.

        Args:
            anglicisms_file: Path to file with list of anglicisms
            anglicism_dict_file: Path to CSV file with anglicism dictionary
            model_name: Name of the language model to use
            semantic_model_name: Name of the semantic similarity model
            device: Device to use for models (None for auto-detection)
        """
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Using device: {self.device}")

        # Initialize NLP components for lemmatization
        print("Initializing NLP components...")
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)

        # Initialize anglicisms data
        self.anglicisms_set = set()
        self.anglicism_dict = {}

        if anglicisms_file:
            self.load_anglicisms_set(anglicisms_file)

        if anglicism_dict_file:
            self.load_anglicism_dictionary(anglicism_dict_file)

        # Load models
        print(f"Loading models: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        # Add special tokens for anglicism replacement
        special_tokens = {"additional_special_tokens": ["<anglicism>", "</anglicism>", "<synonym>", "</synonym>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load semantic model
        print(f"Loading semantic model: {semantic_model_name}")
        self.semantic_model = SentenceTransformer(semantic_model_name, device=self.device)

    def load_anglicisms_set(self, file_path: str) -> None:
        """Load anglicisms from file and convert to lemmas."""
        self.anglicisms_set = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    anglicism = line.strip().lower()
                    # Lemmatize each anglicism
                    anglicism_lemma = self.lemmatize_word(anglicism)
                    self.anglicisms_set.add(anglicism_lemma)

            print(f"Loaded {len(self.anglicisms_set)} anglicisms lemmas from {file_path}")
        except Exception as e:
            print(f"Error loading anglicisms file: {e}")

    def load_anglicism_dictionary(self, file_path: str) -> None:
        """Load dictionary of anglicisms with their synonyms."""
        self.anglicism_dict = {}

        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=',', quotechar='"',
                             on_bad_lines='warn', low_memory=False)

            count_with_synonyms = 0

            for _, row in df.iterrows():
                word = row['word'].strip().lower() if pd.notna(row['word']) else None

                if not word:
                    continue

                synonyms = []

                # Collect non-empty synonyms from columns synonym_1 to synonym_5
                for i in range(1, 6):
                    col_name = f'synonym_{i}'

                    if col_name not in df.columns:
                        continue

                    if pd.notna(row[col_name]) and row[col_name].strip():
                        synonyms.append(row[col_name].strip().lower())

                # Add entry to dictionary if there are synonyms
                if synonyms:
                    self.anglicism_dict[word] = synonyms
                    count_with_synonyms += 1

            print(f"Loaded {count_with_synonyms} anglicisms with synonyms from {len(df)} rows")

        except Exception as e:
            print(f"Error loading anglicism dictionary: {e}")

    @staticmethod
    def ensure_dir(path: str) -> None:
        """Create directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    def load_dataset(self, file_path: str) -> List[Tuple[str, List[str]]]:
        """Load the dataset from CSV file."""
        dataset = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # Clean text from extra quotes
                    text = row[0]
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]

                    try:
                        # Process list of anglicisms
                        anglicisms_str = row[1]
                        anglicisms = json.loads(anglicisms_str)

                        if isinstance(anglicisms, list):
                            dataset.append((text, anglicisms))
                        else:
                            print(f"Warning: Anglicisms not in list format: {anglicisms_str}")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse anglicisms in row: {row}")

        print(f"Successfully loaded {len(dataset)} rows from dataset")
        return dataset

    def lemmatize_word(self, word: str) -> str:
        """Lemmatize a word using Natasha."""
        doc = Doc(word)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        # Return lemma of the first token (for one word there will be only one token)
        if doc.tokens:
            return doc.tokens[0].lemma
        return word  # If lemmatization failed, return original word

    def is_anglicism(self, word: str) -> bool:
        """Check if a word is an anglicism, considering original form and hyphenated parts."""
        word = word.lower().strip()

        # Check original form
        if word in self.anglicisms_set:
            return True

        # Check lemmatized form
        word_lemma = self.lemmatize_word(word)
        if word_lemma in self.anglicisms_set:
            return True

        # If word contains hyphen, check individual parts
        if '-' in word:
            parts = word.split('-')
            for part in parts:
                if part.strip() in self.anglicisms_set:
                    return True

                # Check lemmatized parts
                part_lemma = self.lemmatize_word(part.strip())
                if part_lemma in self.anglicisms_set:
                    return True

        return False

    def extract_quoted_text(self, text: str) -> Set[str]:
        """Extract text in quotes."""
        if pd.isna(text) or not isinstance(text, str):
            return set()

        # Look for text in various types of quotes
        quote_patterns = [
            r'«([^»]+)»',  # «text»
            r'"([^"]+)"',  # "text"
            r'\'([^\']+)\'',  # 'text'
        ]

        quoted_words = set()

        for pattern in quote_patterns:
            try:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Split quoted text into individual words, including hyphenated words
                    words_in_quotes = re.findall(r'\b[а-яА-ЯёЁ]+-?[а-яА-ЯёЁ]*\b', match)
                    quoted_words.update(words_in_quotes)
            except Exception as e:
                print(f"Error processing pattern '{pattern}': {e}")

        return quoted_words

    @staticmethod
    def has_dot_before(text: str, word_pos: int) -> bool:
        """Check if there's a period, exclamation mark, or question mark before a word."""
        # If word is at the beginning of text, there's no dot before it
        if word_pos == 0:
            return False

        # Check characters before word start
        for i in range(word_pos - 1, -1, -1):
            if text[i].isspace():
                continue
            return text[i] in {'.', '!', '?'}

        return False

    def check_text_for_anglicisms(self, text: str,
                                  exceptions_lemmas: Optional[Set[str]] = None,
                                  stopwords_lemmas: Optional[Set[str]] = None) -> List[str]:
        """Check text for anglicisms and return list of found anglicisms."""
        found_anglicisms = []

        # If sets not provided, create empty ones
        if exceptions_lemmas is None:
            exceptions_lemmas = set()
        if stopwords_lemmas is None:
            stopwords_lemmas = set()

        if pd.isna(text) or not isinstance(text, str):
            return []

        # Extract words in quotes
        quoted_words = self.extract_quoted_text(text)
        # Create set of lemmas for words in quotes for fast checking
        quoted_lemmas = {self.lemmatize_word(word.lower()) for word in quoted_words}

        # Split text into words, including hyphenated words
        word_matches = re.finditer(r'\b[а-яА-ЯёЁ]+(?:-[а-яА-ЯёЁ]+)*\b', text)

        for match in word_matches:
            try:
                word = match.group(0)
                word_pos = match.start()

                # Skip words shorter than 4 characters to avoid false positives
                if len(word) <= 3:
                    continue

                # Skip phrases (check for spaces)
                if ' ' in word:
                    continue

                # Check condition: if word starts with capital letter and there's no period before it, it's not an anglicism
                if word[0].isupper() and not self.has_dot_before(text, word_pos):
                    continue

                # Check if word contains digits
                if any(char.isdigit() for char in word):
                    continue

                # Convert to lowercase for further processing
                word_lower = word.lower()

                # Apply lemmatization
                word_lemma = self.lemmatize_word(word_lower)

                # Check that:
                # 1. Word lemma is in anglicisms set
                # 2. Word lemma is NOT in stopwords
                # 3. Word lemma is NOT in exceptions
                # 4. Word is NOT inside quotes (not part of a proper name)
                if (word_lemma in self.anglicisms_set and
                        word_lemma not in stopwords_lemmas and
                        word_lemma not in exceptions_lemmas and
                        word_lemma not in quoted_lemmas):

                    # Additional check if word is part of quoted text
                    is_in_quotes = word.lower() in {q.lower() for q in quoted_words}

                    # If word is not in quotes and not already added, add it
                    if not is_in_quotes and word not in found_anglicisms:
                        found_anglicisms.append(word)
            except Exception as e:
                print(f"Error processing word '{word}': {e}")

        return found_anglicisms

    def generate_synonyms(self, anglicism: str, num_synonyms: int = 7) -> Tuple[List[str], bool]:
        """Find Russian synonyms for anglicism from dictionary."""
        # Convert anglicism to base form
        anglicism_lemma = self.lemmatize_word(anglicism.lower())

        # Set for tracking valid synonyms (to avoid duplicates)
        all_valid_synonyms = set()

        # Flag to track source of synonyms
        from_dictionary = False

        # Check if anglicism is in our dictionary
        if self.anglicism_dict and anglicism_lemma in self.anglicism_dict:
            dictionary_synonyms = self.anglicism_dict[anglicism_lemma]

            for synonym in dictionary_synonyms:
                # Check synonyms from dictionary for validity
                if not self.is_anglicism(synonym):
                    if (self.lemmatize_word(synonym.lower()) != anglicism_lemma and
                            synonym.lower() != anglicism.lower()):
                        all_valid_synonyms.add(synonym)
                        from_dictionary = True

        # Convert set back to list
        final_valid_synonyms = list(all_valid_synonyms)

        # Return all found valid synonyms and source flag (True = from dictionary)
        return final_valid_synonyms, True

    @staticmethod
    def simple_replace_in_text(text: str, anglicism: str, synonym: str) -> str:
        """Simple replacement of anglicism with synonym in text, preserving case."""
        # Find all occurrences of anglicism in text (case-insensitive)
        pattern = re.compile(re.escape(anglicism), re.IGNORECASE)

        # Replace with synonym, preserving case of first letter
        def replace_match(match):
            matched = match.group(0)
            if matched[0].isupper():
                return synonym[0].upper() + synonym[1:]
            return synonym

        return pattern.sub(replace_match, text)

    def generate_combinations_and_replace(self, text: str, anglicisms: List[str],
                                          synonyms_map: Dict[str, List[str]]) -> List[Tuple[str, Dict[str, str]]]:
        """Generate all possible combinations of replacements and return them."""
        # Create list of all possible combinations of synonyms
        synonyms_lists = [synonyms_map[anglicism] for anglicism in anglicisms]
        combinations = list(itertools.product(*synonyms_lists))

        replaced_texts = []

        for combination in combinations:
            current_text = text
            combo_details = {}

            # Apply each replacement in combination
            for i, anglicism in enumerate(anglicisms):
                synonym = combination[i]
                current_text = self.simple_replace_in_text(current_text, anglicism, synonym)
                combo_details[anglicism] = synonym

            replaced_texts.append((current_text, combo_details))

        return replaced_texts

    def transform_sentence_with_synonym(self, anglicism: str, replaced_text: str) -> str:
        """Transform sentence with proper grammar based on replacement example."""
        system_prompt = f"""Ты эксперт по русскому языку. В предложении заменили неуместные слова, но окончания слов могут быть выбраны неправильно.
        Нужно минимально менять предложение, вот пример:
        ДО: "Отметил мэр Игорь Терехов прямой эфир украинского телеканала LIGA."
        ТВОЙ ОТВЕТ: "Отметил мэр Игорь Терехов в прямом эфире украинскому телеканалу LIGA."
        ВАЖНО: Верни только изменённое предложение, без дополнительных объяснений.
        НИ В КОЕМ СЛУЧАЕ НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ СЛОВО "{anglicism}" И ЕГО ПРОИЗВОДНЫЕ.
        Не забывай, что иногда приходится менять соседние слова, чтобы предложение было согласованным."""

        user_prompt = f""" Вот предложение без учета правильных грамматических форм.:
        {replaced_text}. НИ В КОЕМ СЛУЧАЕ НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ СЛОВО "{anglicism}" или его производные."""

        # Format messages according to model format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input data
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.95,
                do_sample=True
            )

        # Extract only generated part
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode response
        transformed_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transformed_text

    def calculate_semantic_similarity(self, original_text: str, replaced_text: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Calculate embeddings for both texts
        embedding1 = self.semantic_model.encode(original_text, convert_to_tensor=True)
        embedding2 = self.semantic_model.encode(replaced_text, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

        return similarity

    def replace_anglicisms(self, text: str, anglicisms: List[str],
                           exceptions_lemmas: Optional[Set[str]] = None,
                           stopwords_lemmas: Optional[Set[str]] = None) -> Tuple[
        Optional[str], Optional[Dict], Optional[str]]:
        """Replace anglicisms in text using dictionary synonyms."""
        original_text = text
        replacement_details = {}

        # Always specify source as dictionary
        source_type = "Dictionary"

        # Step 1: Generate synonyms for each anglicism (from dictionary only)
        synonyms_map = {}
        for anglicism in anglicisms:
            synonyms_result, _ = self.generate_synonyms(anglicism, num_synonyms=5)

            # If no synonyms found for any anglicism, skip sentence
            if not synonyms_result:
                return None, None, None

            # Save only synonym list
            synonyms_map[anglicism] = synonyms_result

        # Step 2: Generate all possible combinations of replacements
        all_replacements = self.generate_combinations_and_replace(text, anglicisms, synonyms_map)

        # Step 3: Calculate semantic similarity for each raw replacement
        replacement_similarities = []
        for replaced_text, combo_details in all_replacements:
            similarity = self.calculate_semantic_similarity(original_text, replaced_text)
            replacement_similarities.append((replaced_text, combo_details, similarity))

        # Step 4: Sort by similarity and take top
        replacement_similarities.sort(key=lambda x: x[2], reverse=True)
        top_replacements = replacement_similarities[:2]  # Number of sentences

        # Step 5: Apply grammatical transformation to each of the top
        transformed_texts = []
        for replaced_text, combo_details, similarity in top_replacements:
            # Transform with proper grammar
            transformed_text = self.transform_sentence_with_synonym(anglicism, replaced_text)

            # Check transformed text for anglicisms
            found_anglicisms = self.check_text_for_anglicisms(
                transformed_text, exceptions_lemmas, stopwords_lemmas
            )

            if found_anglicisms:
                continue  # Skip this option if anglicisms found

            final_similarity = self.calculate_semantic_similarity(original_text, transformed_text)
            transformed_texts.append((transformed_text, combo_details, final_similarity))

        # Step 6: Choose best transformed text
        if not transformed_texts:
            # If all options rejected due to anglicisms
            return None, None, None

        transformed_texts.sort(key=lambda x: x[2], reverse=True)
        best_transformed = transformed_texts[0]

        # Save replacement details
        for anglicism in anglicisms:
            replacement_details[anglicism] = {
                "chosen_synonym": best_transformed[1][anglicism],
                "all_synonyms": synonyms_map[anglicism],
                "similarity": best_transformed[2]
            }

        # Return final text with replacements, replacement details, and source type
        return best_transformed[0], replacement_details, source_type

    def save_batch(self, batch_data: List[Tuple], batch_num: int, output_dir: str) -> str:
        """Save batch of data to CSV file in specified directory."""
        # Create unique filename with batch number
        filename = f"etalon_batch_{batch_num:03d}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            # Write header with added columns
            writer.writerow([
                "Original Text", "Anglicisms", "Replaced Text",
                "Replacement Details", "Semantic Similarity", "Source Type"
            ])

            # Write data
            for row in batch_data:
                writer.writerow(row)

        print(f"Batch {batch_num} saved to {filepath} with {len(batch_data)} rows")
        return filepath

    def process_dataset(self, dataset: List[Tuple[str, List[str]]], output_dir: str,
                        batch_size: int = 10, exceptions_lemmas: Optional[Set[str]] = None,
                        stopwords_lemmas: Optional[Set[str]] = None) -> List[str]:
        """Process dataset in batches and save each batch."""
        current_batch = []
        batch_num = 1
        processed_count = 0
        saved_files = []
        skipped_count = 0  # Counter for skipped sentences

        # Create directory for saving files if it doesn't exist
        self.ensure_dir(output_dir)

        # Initialize progress bar
        progress_bar = tqdm(
            total=len(dataset),
            desc="Processing dataset",
            unit="example",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for i, (text, anglicisms) in enumerate(dataset):
            try:
                # Replace anglicisms with best option
                replaced_text, replacement_details, source_type = self.replace_anglicisms(
                    text, anglicisms, exceptions_lemmas, stopwords_lemmas
                )

                # Check if example was rejected
                if replaced_text is None or replacement_details is None:
                    progress_bar.update(1)
                    skipped_count += 1
                    continue

                # Extract semantic similarity value for best option
                # Take first anglicism to get similarity (all have same value)
                first_anglicism = anglicisms[0]
                similarity = replacement_details[first_anglicism]["similarity"]

                # Save results to current batch
                current_batch.append((
                    text,
                    json.dumps(anglicisms, ensure_ascii=False),
                    replaced_text,
                    json.dumps(replacement_details, ensure_ascii=False),
                    similarity,
                    source_type
                ))

                processed_count += 1

                # If batch size reached or last element, save batch
                if len(current_batch) >= batch_size or i == len(dataset) - 1:
                    if current_batch:  # Check batch is not empty
                        filepath = self.save_batch(current_batch, batch_num, output_dir)
                        saved_files.append(filepath)
                        batch_num += 1
                        current_batch = []  # Reset current batch

            except Exception as e:
                print(f"Error processing item {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()
        print(f"Processing completed. {processed_count} items processed and saved in {batch_num - 1} batches.")
        print(f"Skipped {skipped_count} items (no synonyms in dictionary or other issues).")
        return saved_files

    def process_file(self, input_path: str, output_dir: str, batch_size: int = 10) -> List[str]:
        """Process a file from start to finish."""
        # Load dataset
        print(f"Loading dataset from {input_path}...")
        dataset = self.load_dataset(input_path)
        print(f"Loaded {len(dataset)} examples.")

        # Process dataset
        print("Processing dataset in batches...")
        saved_files = self.process_dataset(
            dataset,
            output_dir,
            batch_size=batch_size
        )

        print(f"Saved results to {len(saved_files)} files in {output_dir}")
        return saved_files


def main():
    # Paths
    input_path = "assets/anglicisms_dataset.csv"
    output_dir = "assets/etalons"
    anglicisms_file = "assets/clean_anglicism_2.txt"
    anglicism_dict_file = "assets/anglicism_dictionary.csv"

    # Create replacer instance
    replacer = AnglicismReplacer(
        anglicisms_file=anglicisms_file,
        anglicism_dict_file=anglicism_dict_file
    )

    # Process file
    replacer.process_file(input_path, output_dir, batch_size=10)


if __name__ == "__main__":
    main()