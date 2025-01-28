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

# Configuration using dataclass
@dataclass
class ChatbotConfig:
    n: int = 5
    vocab_size: int = 50000  # Increased vocabulary size to reduce <unk> occurrences
    max_words: int = 30  # Increased from 25 for longer responses
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

# Personality components
class PersonalityEngine:
    """Advanced personality management with context awareness and intent recognition"""
    
    # Intent patterns for recognition (removed 'fallback' intent)
    INTENT_PATTERNS = {
        'greeting': r'\b(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|howdy)\b',
        'farewell': r'\b(goodbye|bye|see\s*you|farewell|good\s*night)\b',
        'how_are_you': r'\b(how\s*are\s*you|how\s*you\s*doing|how\'?s\s*it\s*going|what\'?s\s*up)\b',
        'thanks': r'\b(thanks|thank\s*you|appreciate|grateful)\b',
        'opinion': r'\b(what\s*(do|should|would)\s*(you|u)\s*(think|feel|believe)|your\s*thoughts)\b',
        'help': r'\b(help|assist|guide|explain|clarify)\b',
        'agreement': r'\b(yes|yeah|sure|okay|alright|correct|right)\b',
        'disagreement': r'\b(no|nope|nah|disagree|incorrect|wrong)\b',
        'hedges': r'\b(perhaps|maybe|I think|it seems|I believe)\b'
    }
    
    # Response templates with context tags
    RESPONSES = {
        'greeting': [
            ("Hi there! How can I help you today?", 'formal'),
            ("Hey! Great to chat with you!", 'casual'),
            ("Hello! I'm ready to assist you.", 'formal'),
            ("Hi! What's on your mind?", 'casual'),
            ("Greetings! How may I help?", 'formal')
        ],
        'farewell': [
            ("Goodbye! Thanks for chatting!", 'casual'),
            ("Take care! Have a great day!", 'casual'),
            ("Farewell! It was nice talking with you.", 'formal'),
            ("See you later! Feel free to return anytime!", 'casual'),
            ("Goodbye! Looking forward to our next conversation.", 'formal')
        ],
        'how_are_you': [
            ("I'm doing well, thanks for asking! How about you?", 'casual'),
            ("I'm great! How's your day going?", 'casual'),
            ("Thanks for asking! I'm here and ready to help. How are you?", 'formal'),
            ("I'm functioning perfectly! How can I assist you today?", 'formal')
        ],
        'thanks': [
            ("You're welcome! Happy to help!", 'casual'),
            ("Glad I could assist!", 'casual'),
            ("My pleasure! Is there anything else you'd like to know?", 'formal'),
            ("You're most welcome! Don't hesitate to ask if you need more help.", 'formal')
        ],
        'opinion': [
            ("Based on what we've discussed, I think...", 'thoughtful'),
            ("From my perspective...", 'thoughtful'),
            ("Here's what I believe about that...", 'thoughtful'),
            ("Let me share my thoughts on this...", 'thoughtful')
        ],
        'help': [
            ("I'll do my best to help. What exactly would you like to know?", 'formal'),
            ("Sure thing! What do you need help with?", 'casual'),
            ("I'd be happy to assist. Could you provide more details?", 'formal'),
            ("Of course! Let me know what you're looking for.", 'casual')
        ],
        'hedges': [
            ("Perhaps, ", 'thoughtful'),
            ("Maybe, ", 'thoughtful'),
            ("I think, ", 'casual'),
            ("It seems, ", 'neutral')
        ],
        'default': [
            ("I'm listening. Tell me more about that.", 'neutral'),
            ("That's interesting! Could you elaborate?", 'neutral'),
            ("I'd like to understand better. Could you explain further?", 'neutral'),
            ("Tell me more about your thoughts on this.", 'neutral')
        ]
    }
    
    # Emotion indicators for response modification
    EMOTIONS = {
        'positive': ["😊", "😄", "👍", "✨"],
        'neutral': ["🤔", "💭", "👋"],
        'thoughtful': ["💡", "🤓", "✍️"]
    }
    
    # Contextual modifiers
    MODIFIERS = {
        'formal': {
            'hedges': ["I believe", "In my view", "It appears that", "Perhaps"],
            'transitions': ["Furthermore", "Moreover", "Additionally", "However"]
        },
        'casual': {
            'hedges': ["I think", "Seems like", "Maybe", "Probably"],
            'transitions': ["Also", "Plus", "But", "Though"]
        },
        'thoughtful': {
            'hedges': ["After consideration", "Upon reflection", "It seems to me", "One might say"],
            'transitions': ["On the other hand", "Nevertheless", "Consequently", "Indeed"]
        }
    }

    def __init__(self):
        """Initialize personality with context tracking"""
        self.conversation_history = []
        self.current_tone = 'neutral'
        self.formality_level = 'casual'
        self.compile_patterns()

    def compile_patterns(self):
        """Compile regex patterns for faster matching"""
        self.compiled_patterns = {
            intent: re.compile(pattern, re.IGNORECASE)
            for intent, pattern in self.INTENT_PATTERNS.items()
        }

    def detect_intent(self, user_input: str) -> str:
        """Detect the user's intent from their input"""
        for intent, pattern in self.compiled_patterns.items():
            if pattern.search(user_input):
                return intent
        return 'default'

    def get_response(self, user_input: str, context: dict = None) -> str:
        """Generate a contextually appropriate response based on user input"""
        intent = self.detect_intent(user_input)
        self.update_conversation_context(user_input, intent)
        
        # Select appropriate response template
        responses = self.RESPONSES.get(intent, self.RESPONSES['default'])
        response, tone = random.choice(responses)
        
        # Add contextual modifications
        response = self.add_contextual_elements(response, tone, intent)
        
        # Update conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'response': response,
            'intent': intent,
            'tone': tone
        })
        
        return response

    def get_intent_response(self, intent: str) -> Optional[str]:
        """Retrieve a response based on a specific intent."""
        responses = self.RESPONSES.get(intent, self.RESPONSES['default'])
        if responses:
            response, tone = random.choice(responses)
            response = self.add_contextual_elements(response, tone, intent)
            return response
        return None

    def add_contextual_elements(self, response: str, tone: str, intent: str) -> str:
        """Add contextual elements to the response"""
        # Add appropriate emotion indicator
        if intent in ['greeting', 'thanks']:
            emotion_type = 'positive'
        elif intent in ['opinion', 'help', 'thoughtful']:
            emotion_type = 'thoughtful'
        else:
            emotion_type = 'neutral'
        emoji = random.choice(self.EMOTIONS[emotion_type])
        
        # Add contextual modifier if appropriate
        if intent in ['opinion', 'help']:
            modifier = random.choice(self.MODIFIERS[tone]['hedges'])
            response = f"{modifier}, {response}"
        
        # Add emoji with reduced probability for formal tone
        if tone != 'formal' or random.random() < 0.3:
            response = f"{response} {emoji}"
        
        return response

    def update_conversation_context(self, user_input: str, intent: str):
        """Update conversation context based on user input and intent"""
        # Adjust formality based on user's style
        if re.search(r'\b(please|kindly|would you|could you)\b', user_input, re.IGNORECASE):
            self.formality_level = 'formal'
        elif re.search(r'\b(hey|hi|yeah|nah|cool|awesome)\b', user_input, re.IGNORECASE):
            self.formality_level = 'casual'
        
        # Track conversation flow
        if len(self.conversation_history) > 3:
            self.conversation_history.pop(0)  # Keep only recent history

# Text Preprocessing
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
                tokens = [token if token in vocab else '<unk>' for token in tokens]
            
            tokens = ['<s>'] + tokens + ['</s>']
            processed_sentences.append(tokens)
        
        result = [token for sentence in processed_sentences for token in sentence]
        self.cache[cache_key] = result
        return result

# N-Gram Model
class NGramModel:
    """Enhanced N-gram model with optimized database operations"""
    
    def __init__(self, config: ChatbotConfig, personality: PersonalityEngine):
        self.config = config
        self.conn = None
        self.preprocessor = TextPreprocessor()
        self.personality = personality  # Store the PersonalityEngine instance
        self.vocab = set()  # Initialize vocabulary
        
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
        self.vocab = {word for word, _ in word_freq.most_common(self.config.vocab_size)}
        self.vocab.add('<unk>')
        
        # Re-process with vocabulary limitation
        tokens = self.preprocessor.preprocess_text(text, vocab=self.vocab)
        
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
            greeting_response = self.personality.get_intent_response('greeting') or "Hello!"
            emoji = random.choice(self.personality.EMOTIONS['neutral'])
            return f"{greeting_response} {emoji}"
        
        # Preprocess the seed text
        seed_tokens = self.preprocessor.preprocess_text(seed, vocab=self.vocab)
        if not seed_tokens:
            generic_prompt = "Could you tell me more about that?"
            return generic_prompt
        
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
            vocab_size = self.config.vocab_size + 1  # Including <unk>
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
                    next_word = random.choice(list(self.vocab))
                
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
            hedge = self.personality.get_intent_response('hedges') or "Perhaps, "
            text = f"{hedge}{text.lower()}"
        
        # Correct spacing before punctuation
        text = re.sub(r'\s([?.!,](?:\s|$))', r'\1', text)
        
        # Capitalize the first letter of the response
        text = text.capitalize()
        
        # Ensure the response ends with proper punctuation
        if text and text[-1] not in {'.', '?', '!'}:
            text += random.choice(['.', '!', '...'])
        
        # Append an emoji with a certain probability to add personality
        if random.random() < self.config.emoji_probability:
            # Choose appropriate emotion category based on intent
            intent = self.personality.detect_intent(seed)
            if intent in ['greeting', 'thanks']:
                emotion_type = 'positive'
            elif intent in ['opinion', 'help']:
                emotion_type = 'thoughtful'
            else:
                emotion_type = 'neutral'
            emoji = random.choice(self.personality.EMOTIONS[emotion_type])
            text += f" {emoji}"
            
        # If the response is too short, prepend a natural filler
        if len(text.split()) < 4:
            text = f"{random.choice(['Hmm... ', 'Well... '])}{text}"
        
        # If the response is still too short or lacks meaningful content, use a generic prompt
        if len(text.split()) <= 2:
            generic_prompt = "Could you tell me more about that?"
            return generic_prompt
        
        return text

# Main Chatbot Class
class Chatbot:
    """Main chatbot class with enhanced interaction capabilities"""
    
    def __init__(self, config: ChatbotConfig = None):
        self.config = config or ChatbotConfig()
        self.personality = PersonalityEngine()  # Instantiate PersonalityEngine
        self.model = NGramModel(self.config, self.personality)  # Pass the instance
    
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
                for book_id, content in zip(book_ids, results):
                    if content:
                        f.write(content + "\n\n")
                        print(f"Book {book_id} downloaded and added to training data.")
                    else:
                        print(f"Failed to download book ID {book_id}.")
    
    def chat(self):
        """Enhanced chat interface with typing animation"""
        print("\nChatbot is ready! Let's chat (type 'exit' to quit):")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                farewell_response = self.personality.get_intent_response('farewell') or "Goodbye!"
                emotion_type = 'neutral'
                # Choose appropriate emotion based on farewell intent
                emoji = random.choice(self.personality.EMOTIONS[emotion_type])
                farewell_message = f"{farewell_response} {emoji}"
                self._animate_typing(farewell_message)
                break
            
            # Detect intent
            intent = self.personality.detect_intent(user_input)
            if intent != 'default':
                response = self.personality.get_intent_response(intent) or "I'm not sure how to respond to that."
            else:
                # Generate response via n-gram model for default intent
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

# Entry Point
async def main():
    """Enhanced main function with async support"""
    try:
        # Initialize chatbot with custom configuration
        config = ChatbotConfig(
            n=5,  # Use 5-grams for better context
            vocab_size=50000,  # Increased vocabulary size to reduce <unk> occurrences
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
