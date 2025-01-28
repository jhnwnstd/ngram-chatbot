import re
import nltk
import contractions
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
import random
import os
from urllib.parse import urljoin
import time
import sqlite3
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)

# Ensure all necessary NLTK data packages are downloaded
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])

# Personality components
class PersonalityEngine:
    """Enhanced personality management for more natural responses"""
    
    GREETINGS = [
        "Hi there!", "Hello!", "Hey!", "Greetings!", 
        "Good to see you!", "Nice to meet you!", "Hi!"
    ]
    FAREWELLS = [
        "Goodbye!", "See you later!", "Farewell!", "Bye!", 
        "Take care!", "Until next time!", "Catch you later!"
    ]
    EMOJIS = ["😊", "🤔", "😄", "🙂", "👋", "✨", "💭"]
    HEDGES = [
        "Well... ", "Hmm... ", "You know... ", "I think... ",
        "Let me see... ", "It seems... ", "Perhaps... "
    ]
    FALLBACK_RESPONSES = [
        "I'm not sure I understand. Could you tell me more?",
        "Interesting! Can you elaborate on that?",
        "That's intriguing. Could you explain further?",
        "I'd love to hear more about that.",
        "Could you rephrase that for me?"
    ]
    
    @classmethod
    def get_response_enhancement(cls, response_type: str) -> str:
        """Get a random enhancement of specified type with weighted probabilities"""
        if response_type == "greeting":
            return random.choices(
                cls.GREETINGS, 
                weights=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
            )[0]
        elif response_type == "farewell":
            return random.choices(
                cls.FAREWELLS, 
                weights=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
            )[0]
        elif response_type == "emoji":
            return random.choice(cls.EMOJIS)
        elif response_type == "hedges":
            return random.choice(cls.HEDGES)
        elif response_type == "fallback":
            return random.choice(cls.FALLBACK_RESPONSES)
        return ""

# Configuration using dataclass
@dataclass
class ChatbotConfig:
    n: int = 5
    vocab_size: int = 25000
    max_words: int = 25
    temperature: float = 0.75
    max_repeats: int = 2
    batch_size: int = 100000
    db_file: str = 'ngram_model.db'
    training_file: str = 'train.txt'
    min_response_length: int = 10
    max_response_length: int = 25
    hesitation_probability: float = 0.05
    hedge_probability: float = 0.3
    emoji_probability: float = 0.2

class TextPreprocessor:
    """Enhanced text preprocessing with caching and optimization"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {
            'oh', 'well', 'hey', 'please', 'thanks', 'okay', 'yes', 'no'
        }
        self.cache = {}
        
    def preprocess_text(self, text: str, vocab: Optional[set] = None) -> List[str]:
        """Process text with caching and optimized preprocessing"""
        cache_key = hash(text + str(vocab))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Remove Project Gutenberg metadata
        text = re.sub(r'\*+\s*END.*?(\*+\s*$)?', '', text, flags=re.DOTALL)
        text = text.lower()
        text = contractions.fix(text)
        
        sentences = nltk.sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            # Optimized regex for punctuation
            sentence = re.sub(r'[^a-z?.!,¿\']+', ' ', sentence)
            tokens = nltk.word_tokenize(sentence)
            
            # Batch lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
            
            if vocab is not None:
                tokens = [token if token in vocab else '<UNK>' for token in tokens]
            
            tokens = ['<s>'] + tokens + ['</s>']
            processed_sentences.append(tokens)
        
        result = [token for sentence in processed_sentences for token in sentence]
        self.cache[cache_key] = result
        return result

class NGramModel:
    """Enhanced N-gram model with optimized database operations"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.conn = None
        self.preprocessor = TextPreprocessor()
        
    async def download_gutenberg_text(self, book_id: str) -> Optional[str]:
        """Asynchronously download text from Project Gutenberg"""
        base_url = "https://www.gutenberg.org/cache/epub/"
        book_url = urljoin(base_url, f"{book_id}/pg{book_id}.txt")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(book_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Clean content
                        content = re.sub(
                            r'^.*?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\n',
                            '', 
                            content, 
                            flags=re.DOTALL|re.IGNORECASE
                        )
                        content = re.sub(
                            r'END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?$',
                            '', 
                            content, 
                            flags=re.DOTALL|re.IGNORECASE
                        )
                        # Remove control characters
                        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
                        return content.strip()
                    return None
            except Exception as e:
                logging.error(f"Failed to download book {book_id}: {str(e)}")
                return None

    def setup_database(self):
        """Set up SQLite database with optimized settings"""
        self.conn = sqlite3.connect(self.config.db_file)
        cursor = self.conn.cursor()
        
        # Optimize SQLite settings
        cursor.executescript('''
            PRAGMA synchronous = OFF;
            PRAGMA journal_mode = MEMORY;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -2000000;  -- Use 2GB of cache
            
            CREATE TABLE IF NOT EXISTS ngrams (
                prefix TEXT,
                next_word TEXT,
                count INTEGER,
                PRIMARY KEY (prefix, next_word)
            ) WITHOUT ROWID;  -- Optimize for faster lookups
        ''')
        self.conn.commit()

    def build_model(self, text: str):
        """Build n-gram model with progress tracking and optimization"""
        logging.info("Building n-gram model...")
        
        # Count word frequencies and build vocabulary
        tokens = self.preprocessor.preprocess_text(text)
        word_freq = Counter(tokens)
        vocab = {word for word, _ in word_freq.most_common(self.config.vocab_size)}
        vocab.add('<UNK>')
        
        # Re-process with vocabulary limitation
        tokens = self.preprocessor.preprocess_text(text, vocab=vocab)
        
        # Generate and store n-grams with progress tracking
        ngram_generator = ngrams(tokens, self.config.n)
        total_tokens = len(tokens)
        expected_ngrams = total_tokens - self.config.n + 1
        
        with tqdm.tqdm(total=expected_ngrams, desc="Building n-grams") as pbar:
            ngram_batch = []
            
            for gram in ngram_generator:
                prefix = ' '.join(gram[:-1])
                next_word = gram[-1]
                ngram_batch.append((prefix, next_word))
                
                if len(ngram_batch) >= self.config.batch_size:
                    self._store_ngram_batch(ngram_batch)
                    pbar.update(len(ngram_batch))
                    ngram_batch = []
            
            if ngram_batch:
                self._store_ngram_batch(ngram_batch)
                pbar.update(len(ngram_batch))

    def _store_ngram_batch(self, batch: List[Tuple[str, str]]):
        """Efficiently store n-gram batch in database"""
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO ngrams (prefix, next_word, count)
            VALUES (?, ?, 1)
            ON CONFLICT(prefix, next_word) DO UPDATE SET count = count + 1
        ''', batch)
        self.conn.commit()

    def prefix_exists(self, cursor: sqlite3.Cursor, prefix: str) -> bool:
        """Check if a prefix exists in the ngram database."""
        try:
            cursor.execute('SELECT 1 FROM ngrams WHERE prefix = ? LIMIT 1', (prefix,))
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logging.error(f"Database error in prefix_exists: {e}")
            return False

    def generate_response(
        self, 
        seed: str, 
        n: int, 
        temperature: float, 
        max_words: int, 
        max_repeats: int
    ) -> str:
        """
        Generate an enhanced and coherent response using the n-gram model.
        
        Parameters:
            seed (str): The input text from the user.
            n (int): The order of the n-gram model.
            temperature (float): Controls the randomness of the predictions.
            max_words (int): Maximum number of words in the response.
            max_repeats (int): Maximum number of allowed repetitions for words/bigrams.
        
        Returns:
            str: The generated response.
        """
        if not seed.strip():
            return f"{PersonalityEngine.get_response_enhancement('greeting')} {PersonalityEngine.get_response_enhancement('emoji')}"
        
        # Preprocess the seed text
        seed_tokens = self.preprocessor.preprocess_text(seed)
        if not seed_tokens:
            return PersonalityEngine.get_response_enhancement('fallback')
        
        response = seed_tokens.copy()
        
        # Determine the desired response length
        response_length = random.randint(self.config.min_response_length, self.config.max_response_length)
        
        # Tracking structures to prevent repetition
        recent_words = []
        word_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        
        cursor = self.conn.cursor()
        
        for _ in range(response_length):
            # Determine the current prefix based on n-gram order
            prefix_length = n - 1
            prefix_tokens = response[-prefix_length:] if len(response) >= prefix_length else response[:]
            prefix = ' '.join(prefix_tokens)
            
            # Dynamically adjust temperature to balance creativity and coherence
            current_temp = max(0.5, temperature - (len(response) / 100))  # Adjusted denominator for smoother decay
            
            # Implement a back-off mechanism to handle unseen prefixes
            fallback_n = n
            fallback_prefix = prefix
            
            while not self.prefix_exists(cursor, fallback_prefix) and fallback_n > 1:
                fallback_n -= 1
                fallback_prefix_length = fallback_n - 1
                fallback_prefix_tokens = response[-fallback_prefix_length:] if len(response) >= fallback_prefix_length else response[:]
                fallback_prefix = ' '.join(fallback_prefix_tokens)
                current_temp += 0.1  # Increase creativity slightly with each fallback
            
            if not self.prefix_exists(cursor, fallback_prefix):
                break  # No suitable prefix found; terminate response generation
            
            # Retrieve possible next words and their counts
            try:
                cursor.execute('SELECT next_word, count FROM ngrams WHERE prefix = ?', (fallback_prefix,))
                results = cursor.fetchall()
            except sqlite3.Error as e:
                logging.error(f"Database error in generate_response: {e}")
                break
            
            if not results:
                break  # No continuations found for the current prefix
            
            choices, counts = zip(*results)
            
            # Apply Add-One (Laplace) Smoothing to handle unseen words
            total_count = sum(counts)
            vocab_size = self.config.vocab_size + 1  # Including <UNK>
            smoothed_probabilities = [(count + 1) / (total_count + vocab_size) for count in counts]
            
            # Normalize the probabilities
            total_prob = sum(smoothed_probabilities)
            normalized_probabilities = [prob / total_prob for prob in smoothed_probabilities]
            
            # Apply temperature scaling for randomness control
            if current_temp != 1.0:
                adjusted_probabilities = [prob ** (1.0 / current_temp) for prob in normalized_probabilities]
                # Re-normalize after temperature adjustment
                total_adj_prob = sum(adjusted_probabilities)
                normalized_probabilities = [prob / total_adj_prob for prob in adjusted_probabilities]
            
            # Filter out words that exceed repetition thresholds
            filtered_choices = [
                word for word, prob in zip(choices, normalized_probabilities)
                if word not in recent_words and word_counts[word] < max_repeats
            ]
            filtered_probs = [
                prob for word, prob in zip(choices, normalized_probabilities)
                if word not in recent_words and word_counts[word] < max_repeats
            ]
            
            # If no words are left after filtering, reset filters
            if not filtered_choices:
                filtered_choices = choices
                filtered_probs = normalized_probabilities
            
            # Sample the next word based on the filtered probabilities
            try:
                next_word = random.choices(filtered_choices, weights=filtered_probs, k=1)[0]
                
                # Replace <unk> with a random word from the vocabulary to reduce incoherence
                if next_word == '<unk>':
                    # Option 1: Choose a random word from the known vocabulary
                    # Assuming self.vocab is a set containing all known words
                    if hasattr(self, 'vocab') and self.vocab:
                        next_word = random.choice(list(self.vocab))
                    else:
                        # Fallback if vocab is not defined
                        next_word = random.choice(filtered_choices)
                
                # Prevent specific bigram repetitions
                if len(response) >= 1:
                    bigram = f"{response[-1]} {next_word}"
                    bigram_counts[bigram] += 1
                    if bigram_counts[bigram] > max_repeats:
                        continue  # Skip adding this bigram to avoid repetition
                
                # Prevent word overuse
                if word_counts[next_word] >= max_repeats:
                    continue  # Skip adding this word to avoid repetition
                
                # Append the next word to the response
                response.append(next_word)
                word_counts[next_word] += 1
                
                # Update recent words tracking
                recent_words.append(next_word)
                if len(recent_words) > 5:  # Window size for recent words
                    removed_word = recent_words.pop(0)
                    word_counts[removed_word] = max(word_counts[removed_word] - 1, 0)
                
                # Introduce natural hesitations with a small probability
                if random.random() < self.config.hesitation_probability:
                    hesitation = random.choice(["um", "like", "you know"])
                    response.append(hesitation)
                    word_counts[hesitation] += 1
                    
            except Exception as e:
                logging.error(f"Error during response generation: {e}")
                break
            
            # Implement natural stopping points based on punctuation
            if len(response) > 10 and next_word in {'.', '?', '!'}:
                if random.random() < 0.7:
                    break  # 70% chance to end the response here
            
            # Prevent over-generation by enforcing maximum word limit
            if len(response) >= max_words:
                break
        
        cursor.close()  # Close the database cursor
        
        # Post-processing the generated response
        # Remove special tokens like <s> and </s>
        filtered_response = [word for word in response if word not in {'<s>', '</s>'}]
        text = ' '.join(filtered_response)
        
        # Add hedging phrases to enhance personality
        if random.random() < self.config.hedge_probability:
            text = f"{PersonalityEngine.get_response_enhancement('hedges')}{text.lower()}"
        
        # Correct spacing before punctuation
        text = re.sub(r'\s([?.!,](?:\s|$))', r'\1', text)
        
        # Capitalize the first letter of the response
        text = text.capitalize()
        
        # Ensure the response ends with proper punctuation
        if text and text[-1] not in {'.', '?', '!'}:
            text += random.choice(['.', '!', '...'])
        
        # Append an emoji with a certain probability to add personality
        if random.random() < self.config.emoji_probability:
            text += f" {PersonalityEngine.get_response_enhancement('emoji')}"
            
        # If the response is too short, prepend a natural filler
        if len(text.split()) < 4:
            text = f"{random.choice(['Hmm... ', 'Well... '])}{text}"
        
        # If the response is still too short or lacks meaningful content, use a fallback response
        if len(text.split()) <= 2:
            return PersonalityEngine.get_response_enhancement('fallback')
        
        return text

class Chatbot:
    """Main chatbot class with enhanced interaction capabilities"""
    
    def __init__(self, config: ChatbotConfig = None):
        self.config = config or ChatbotConfig()
        self.model = NGramModel(self.config)
        
    async def setup(self):
        """Set up chatbot with enhanced initialization"""
        if not os.path.exists(self.config.training_file):
            await self._initialize_training_data()
        
        self.model.setup_database()
        
        with open(self.config.training_file, 'r', encoding='utf-8') as f:
            training_text = f.read()
            
        if training_text.strip():
            self.model.build_model(training_text)
        else:
            raise ValueError("No training data available")
    
    async def _initialize_training_data(self):
        """Initialize training data with parallel downloads"""
        print("No training data found. Downloading from Project Gutenberg...")
        book_ids = []
        
        while True:
            book_id = input("Enter Project Gutenberg book ID (or 'done' to finish): ").strip()
            if book_id.lower() == 'done':
                break
            if book_id.isdigit():
                book_ids.append(book_id)
            else:
                print("Please enter a numeric book ID.")
        
        if not book_ids:
            raise ValueError("No books selected for training")
        
        # Download books in parallel
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.model.download_gutenberg_text(book_id) 
                for book_id in book_ids
            ]
            results = await asyncio.gather(*tasks)
            
            with open(self.config.training_file, 'w', encoding='utf-8') as f:
                for content in results:
                    if content:
                        f.write(content + "\n\n")
                        print(f"Book downloaded and added to training data.")
                    else:
                        print(f"Failed to download book ID {book_id}.")
    
    def chat(self):
        """Enhanced chat interface with typing animation"""
        print("\nChatbot is ready! Let's chat (type 'exit' to quit):")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                farewell = f"{PersonalityEngine.get_response_enhancement('farewell')} {PersonalityEngine.get_response_enhancement('emoji')}"
                self._animate_typing(farewell)
                break
            
            # Generate and display response
            response = self.model.generate_response(
                seed=user_input, 
                n=self.config.n, 
                temperature=random.uniform(0.6, 0.9),
                max_words=random.randint(15, self.config.max_words),
                max_repeats=self.config.max_repeats
            )
            self._animate_typing(response, prefix="Chatbot: ")
    
    def _animate_typing(self, text: str, prefix: str = "", typing_speed: tuple = (0.02, 0.05)):
        """Animate typing with realistic variations in speed"""
        print(prefix, end='', flush=True)
        
        # Add natural typing variations
        for char in text:
            # Slow down for punctuation
            if char in '.!?,':
                time.sleep(random.uniform(0.1, 0.2))
            # Pause briefly at spaces
            elif char == ' ':
                time.sleep(random.uniform(0.03, 0.07))
            # Normal typing speed with slight variations
            else:
                time.sleep(random.uniform(*typing_speed))
            print(char, end='', flush=True)
        print()

async def main():
    """Enhanced main function with async support"""
    try:
        # Initialize chatbot with custom configuration
        config = ChatbotConfig(
            n=5,  # Use 5-grams for better context
            vocab_size=25000,  # Increased vocabulary size
            max_words=30,  # Allow longer responses
            temperature=0.75,  # Balanced creativity
            max_repeats=2,  # Prevent repetition
            batch_size=100000,  # Efficient batch processing
            db_file='ngram_model.db',
            training_file='train.txt',
            min_response_length=10,
            max_response_length=25,
            hesitation_probability=0.05,
            hedge_probability=0.3,
            emoji_probability=0.2
        )
        
        chatbot = Chatbot(config)
        await chatbot.setup()
        
        # Optionally, allow adding more books after initial setup
        print("\nCurrent training data size:", os.path.getsize(config.training_file), "bytes")
        print("Would you like to add more books? (y/n)")
        if input("> ").lower() == 'y':
            while True:
                book_id = input("Enter Project Gutenberg book ID (or 'done' to finish): ").strip()
                if book_id.lower() == 'done':
                    break
                if book_id.isdigit():
                    content = await chatbot.model.download_gutenberg_text(book_id)
                    if content:
                        with open(config.training_file, 'a', encoding='utf-8') as f:
                            f.write(content + "\n\n")
                        print(f"Book {book_id} added to training data!")
                        # Rebuild the model with the new data
                        chatbot.model.build_model(content)
                    else:
                        print("Invalid book ID or download failed. Try again.")
                else:
                    print("Please enter a numeric book ID.")
        
        # Start chat interface
        chatbot.chat()
        
    except KeyboardInterrupt:
        print("\nGoodbye! Thanks for chatting! 👋")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print("\nAn error occurred. Please check the logs for details.")
    finally:
        # Clean up resources
        if 'chatbot' in locals() and chatbot.model.conn:
            chatbot.model.conn.close()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())