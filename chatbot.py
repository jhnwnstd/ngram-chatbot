import re
import nltk
import contractions
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
import random
import sys
import os
import requests
from urllib.parse import urljoin
import time
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure all necessary NLTK data packages are downloaded
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])

# Personality components
GREETINGS = ["Hi there!", "Hello!", "Hey!", "Greetings!"]
FAREWELLS = ["Goodbye!", "See you later!", "Farewell!", "Bye!"]
EMOJIS = ["😊", "🤔", "😄", "🙂"]
HEDGES = ["Well... ", "Hmm... ", "You know... ", "I think... "]
FALLBACK_RESPONSES = [
    "I'm not sure I understand. Could you tell me more?",
    "Interesting! Can you elaborate on that?",
    "Hmm... that's something to think about.",
    "Can you explain that a bit more?",
    "Let's talk more about that!"
]

# SQLite database file
DB_FILE = 'ngram_model.db'

def preprocess_text(text, vocab=None):
    """Process text to make it suitable for n-gram modeling with sentence markers."""
    # Remove Project Gutenberg metadata
    text = re.sub(r'\*+\s*END.*?(\*+\s*$)?', '', text, flags=re.DOTALL)
    
    # Lowercase the text
    text = text.lower()
    
    # Expand contractions using the contractions library
    text = contractions.fix(text)
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    processed_sentences = []
    
    for sentence in sentences:
        # Keep conversational punctuation
        sentence = re.sub(r'[^a-z?.!,¿\'’]+', ' ', sentence)
        
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)
        
        # Lemmatize tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Define discourse markers to keep
        keep_words = {'oh', 'well', 'hey', 'please', 'thanks', 'okay', 'yes', 'no'}
        
        # Define stop words, excluding important discourse markers
        stop_words = set(stopwords.words('english')) - keep_words
        
        # Remove stop words and single-character tokens
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        # Replace rare words with <UNK> if vocabulary is provided
        if vocab is not None:
            tokens = [token if token in vocab else '<UNK>' for token in tokens]
        
        # Add start and end markers
        tokens = ['<s>'] + tokens + ['</s>']
        
        processed_sentences.append(tokens)
    
    # Flatten the list of tokens
    tokens = [token for sentence in processed_sentences for token in sentence]
    
    return tokens

def download_gutenberg_text(book_id):
    """Download and clean text from Project Gutenberg."""
    base_url = "https://www.gutenberg.org/cache/epub/"
    book_url = urljoin(base_url, f"{book_id}/pg{book_id}.txt")
    
    try:
        response = requests.get(book_url)
        response.raise_for_status()
        
        # Remove Project Gutenberg headers and footers
        content = re.sub(r'^.*?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\n', '', 
                        response.text, flags=re.DOTALL|re.IGNORECASE)
        content = re.sub(r'END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?$', '', 
                        content, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove control characters
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        
        return content.strip()
        
    except Exception as e:
        logging.error(f"Failed to download book {book_id}: {str(e)}")
        return None

def generate_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return ngrams(tokens, n)

def build_ngram_model(text, n, top_k=15000):
    """Build an n-gram model using SQLite for storage."""
    logging.info("Connecting to SQLite database...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Optimize SQLite for faster inserts
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')
    cursor.execute('PRAGMA temp_store = MEMORY')
    
    # Create table for n-grams without indexes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ngrams (
            prefix TEXT,
            next_word TEXT,
            count INTEGER,
            PRIMARY KEY (prefix, next_word)
        )
    ''')
    conn.commit()

    # Count word frequencies
    logging.info("Counting word frequencies for vocabulary limitation...")
    tokens = preprocess_text(text)
    word_freq = Counter(tokens)
    most_common = word_freq.most_common(top_k)
    vocab = set([word for word, freq in most_common])
    vocab.add('<UNK>')  # Add <UNK> token
    logging.info(f"Top {top_k} words selected for vocabulary.")
    
    # Re-process tokens with vocabulary limitation
    tokens = preprocess_text(text, vocab=vocab)
    logging.info("Preprocessing completed with vocabulary limitation.")

    total_ngrams = 0
    batch_size = 100000  # Number of n-grams per batch
    ngram_batch = []

    logging.info("Starting n-gram generation and storage...")

    # Generate n-grams using a generator to save memory
    for gram in generate_ngrams(tokens, n):
        prefix = ' '.join(gram[:-1])
        next_word = gram[-1]
        ngram_batch.append((prefix, next_word))
        total_ngrams += 1

        if total_ngrams % batch_size == 0:
            cursor.executemany('''
                INSERT INTO ngrams (prefix, next_word, count)
                VALUES (?, ?, 1)
                ON CONFLICT(prefix, next_word) DO UPDATE SET count = count + 1
            ''', ngram_batch)
            conn.commit()
            logging.info(f"Processed {total_ngrams} n-grams...")
            ngram_batch = []

    # Insert any remaining n-grams
    if ngram_batch:
        cursor.executemany('''
            INSERT INTO ngrams (prefix, next_word, count)
            VALUES (?, ?, 1)
            ON CONFLICT(prefix, next_word) DO UPDATE SET count = count + 1
        ''', ngram_batch)
        conn.commit()
        logging.info(f"Processed {total_ngrams} n-grams...")

    # Create index on prefix after all inserts for faster querying
    logging.info("Creating index on prefix...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_prefix ON ngrams(prefix)')
    conn.commit()

    # Close the database connection
    conn.close()

    logging.info(f"Total n-grams processed: {total_ngrams}")

def prefix_exists(cursor, prefix):
    """Check if a prefix exists in the ngram database."""
    try:
        cursor.execute('SELECT 1 FROM ngrams WHERE prefix = ? LIMIT 1', (prefix,))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logging.error(f"Database error in prefix_exists: {e}")
        return False

def generate_response(ngram_model_conn, seed, n, 
                     max_words=25, temperature=0.75, 
                     max_repeats=2):
    """Generate a human-like response based on the n-gram model with Add-One smoothing."""
    if not seed.strip():
        return f"{random.choice(GREETINGS)} {random.choice(EMOJIS)}"
    
    seed_tokens = preprocess_text(seed)
    response = seed_tokens.copy()
    
    response_length = random.randint(10, max_words)  # Vary response length
    
    recent_words = []  # To track recently used words
    word_counts = defaultdict(int)  # To track word repetitions
    bigram_counts = defaultdict(int)  # To track bigram repetitions
    
    cursor = ngram_model_conn.cursor()
    
    for _ in range(response_length):
        prefix_length = n - 1
        if len(response) >= prefix_length:
            prefix_tokens = response[-prefix_length:]
        else:
            prefix_tokens = response[:]
        prefix = ' '.join(prefix_tokens)
        
        # Adjust temperature dynamically (optional)
        current_temp = max(0.5, temperature - (len(response)/50))
        
        # Fallback to smaller n-grams if prefix not found
        temp_fallback_n = n
        temp_prefix = prefix
        
        while not prefix_exists(cursor, temp_prefix) and temp_fallback_n > 1:
            temp_fallback_n -= 1
            temp_prefix_length = temp_fallback_n - 1
            if len(response) >= temp_prefix_length:
                temp_prefix_tokens = response[-temp_prefix_length:]
            else:
                temp_prefix_tokens = response[:]
            temp_prefix = ' '.join(temp_prefix_tokens)
            current_temp += 0.1  # Increase creativity
        
        if not prefix_exists(cursor, temp_prefix):
            break  # Unable to find a suitable prefix
        
        # Fetch next words and counts using the existing cursor
        try:
            cursor.execute('SELECT next_word, count FROM ngrams WHERE prefix = ?', (temp_prefix,))
            results = cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database error in generate_response: {e}")
            break
        
        if not results:
            break  # No available next words
        
        choices, counts = zip(*results)
        
        # Apply Add-One Smoothing: P(next_word | prefix) = (count + 1) / (total_count + V)
        total_count = sum(counts)
        vocab_size = 15001  # As per top_k=15000 + <UNK>
        
        probabilities = [(count + 1) / (total_count + vocab_size) for count in counts]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
        
        # Apply temperature scaling (optional)
        if current_temp != 1.0:
            probabilities = [prob ** (1.0 / current_temp) for prob in probabilities]
            # Re-normalize
            total_prob = sum(probabilities)
            probabilities = [prob / total_prob for prob in probabilities]
        
        # Exclude words exceeding repetition threshold
        filtered_choices = [word for word, prob in zip(choices, probabilities) 
                            if word not in recent_words and word_counts[word] < max_repeats]
        filtered_probabilities = [prob for word, prob in zip(choices, probabilities) 
                                  if word not in recent_words and word_counts[word] < max_repeats]
        
        if not filtered_choices:
            # Reset filters if all choices are excluded
            filtered_choices = choices
            filtered_probabilities = probabilities
        
        try:
            next_word = random.choices(filtered_choices, weights=filtered_probabilities, k=1)[0]
            
            # Prevent specific phrase repetitions using bigram counts
            if len(response) >= 1:
                bigram = ' '.join([response[-1], next_word])
                bigram_counts[bigram] += 1
                if bigram_counts[bigram] > max_repeats:
                    continue  # Skip adding this bigram to avoid repetition
            
            # Prevent word overuse
            if word_counts[next_word] >= max_repeats:
                continue  # Skip adding this word to avoid repetition
            
            response.append(next_word)
            word_counts[next_word] += 1
            
            # Update recent words
            recent_words.append(next_word)
            if len(recent_words) > 5:  # Adjust the window size as needed
                removed_word = recent_words.pop(0)
                word_counts[removed_word] = max(word_counts[removed_word] - 1, 0)
            
            # Add natural hesitation with reduced probability
            if random.random() < 0.05:
                hesitation = random.choice(["um", "like", "you know"])
                response.append(hesitation)
                word_counts[hesitation] += 1
                
        except Exception as e:
            logging.error(f"Error during response generation: {e}")
            break
        
        # Natural stopping points
        if len(response) > 10 and next_word in {'.', '?', '!'}:
            if random.random() < 0.7:
                break
        # Additional stopping criteria to prevent over-generation
        if len(response) >= max_words:
            break

    cursor.close()  # Close the cursor after response generation
    
    # Post-processing
    # Exclude start and end tokens
    filtered_response = [word for word in response if word not in {'<s>', '</s>'}]
    text = ' '.join(filtered_response)
    
    # Add hedging phrases
    if random.random() < 0.3:
        text = random.choice(HEDGES) + text.lower()
    
    # Capitalization and punctuation
    text = re.sub(r'\s([?.!,](?:\s|$))', r'\1', text)
    text = text.capitalize()
    
    # Ensure final punctuation
    if text and text[-1] not in {'.', '?', '!'}:
        text += random.choice(['.', '!', '...'])
    
    # Add personality
    if random.random() < 0.2:
        text += " " + random.choice(EMOJIS)
        
    if len(text.split()) < 4:
        text = random.choice(["Hmm... ", "Well... "]) + text
    
    # Fallback response if no meaningful content was generated
    if len(text.split()) <= 2:
        return random.choice(FALLBACK_RESPONSES)
    
    return text

def main():
    print("Welcome to the N-Gram Chatbot! 👋")
    
    # Check for existing training file
    training_text = ""
    if os.path.exists('train.txt'):
        with open('train.txt', 'r', encoding='utf-8') as f:
            training_text = f.read()
    
    # Book download flow if no training data
    if not training_text.strip():
        print("No training data found. Would you like to download from Project Gutenberg? (y/n)")
        if input("> ").lower() == 'y':
            while True:
                book_id = input("Enter Project Gutenberg book ID (or 'done' to finish): ").strip()
                if book_id.lower() == 'done':
                    break
                
                if book_id.isdigit():
                    content = download_gutenberg_text(book_id)
                    if content:
                        with open('train.txt', 'a', encoding='utf-8') as f:
                            f.write(content + "\n\n")
                        print(f"Book {book_id} added to training data!")
                    else:
                        print("Invalid book ID or download failed. Try again.")
                else:
                    print("Please enter a numeric book ID.")
    
    # Load final training text
    try:
        with open('train.txt', 'r', encoding='utf-8') as f:
            training_text = f.read()
    except FileNotFoundError:
        print("Error: Training file not found after setup.")
        sys.exit(1)
    
    if not training_text.strip():
        print("Error: No training data available.")
        sys.exit(1)
    
    # Option to add more books
    print("\nCurrent training data size:", len(training_text), "characters")
    print("Would you like to add more books? (y/n)")
    if input("> ").lower() == 'y':
        while True:
            book_id = input("Enter Project Gutenberg book ID (or 'done' to finish): ").strip()
            if book_id.lower() == 'done':
                break
            
            if book_id.isdigit():
                content = download_gutenberg_text(book_id)
                if content:
                    with open('train.txt', 'a', encoding='utf-8') as f:
                        f.write(content + "\n\n")
                    print(f"Book {book_id} added to training data!")
                else:
                    print("Invalid book ID or download failed. Try again.")
            else:
                print("Please enter a numeric book ID.")
        
        # Reload training text after adding new books
        try:
            with open('train.txt', 'r', encoding='utf-8') as f:
                training_text = f.read()
        except FileNotFoundError:
            print("Error: Training file not found after adding books.")
            sys.exit(1)
    
    # Build n-gram model
    n = 5  # Tetragrams
    top_k = 25000  # Increased vocabulary size
    logging.info("Building n-gram model. This may take a while...")
    build_ngram_model(training_text, n, top_k=top_k)
    logging.info("n-gram model built successfully.")
    
    # Connect to SQLite database for querying
    conn = sqlite3.connect(DB_FILE)
    
    # Chat interface
    print("\nChatbot is ready! Let's chat (type 'exit' to quit):")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print(f"{random.choice(FAREWELLS)} {random.choice(EMOJIS)}")
            break
        
        # Generate chatbot response
        response = generate_response(
            ngram_model_conn=conn, 
            seed=user_input, 
            n=n, 
            temperature=random.uniform(0.6, 0.9),
            max_words=random.randint(15, 25),
            max_repeats=2
        )
        
        # Simulate typing animation
        print("Chatbot: ", end='', flush=True)
        for char in response:
            print(char, end='', flush=True)
            time.sleep(random.uniform(0.02, 0.05))
        print()
    
    # Close the database connection before exiting
    conn.close()

if __name__ == "__main__":
    main()