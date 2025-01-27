# N-Gram Chatbot

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Initial Setup](#initial-setup)
  - [Training the Model](#training-the-model)
  - [Starting the Chatbot](#starting-the-chatbot)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

The **N-Gram Chatbot** is a simple conversational agent built using Python's n-gram modeling approach. Leveraging SQLite for efficient data storage and retrieval, this chatbot can engage in basic conversations by predicting the next word based on preceding word sequences. It's a project for those interested in natural language processing (NLP) and understanding the fundamentals of language models.

## Features

- **N-Gram Model:** Utilizes trigrams (or higher-order n-grams) to predict and generate responses.
- **SQLite Integration:** Efficient storage and querying of n-gram data using SQLite.
- **Dynamic Vocabulary Management:** Limits vocabulary to the top 15,000 words for optimal performance, replacing rare words with `<UNK>`.
- **Personality Components:** Incorporates greetings, farewells, emojis, hedging phrases, and fallback responses to make interactions feel more natural.
- **Hesitation Markers:** Adds natural pauses like "um" or "you know" to simulate human-like conversation.
- **Repetition Prevention:** Implements mechanisms to avoid repetitive responses and overuse of specific words or phrases.
- **User-Friendly Interface:** Simple command-line interface for easy interaction.
- **Project Gutenberg Integration:** Allows users to download and use books from Project Gutenberg as training data.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: Python package manager. It typically comes bundled with Python.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jhnwnstd/ngram-chatbot.git
   cd ngram-chatbot
   ```

2. **Create a Virtual Environment (Recommended)**

   It's good practice to use a virtual environment to manage dependencies.

   ```bash
   python -m venv chatbot_env
   ```

3. **Activate the Virtual Environment**

   - **Windows:**

     ```bash
     chatbot_env\Scripts\activate
     ```

   - **macOS/Linux:**

     ```bash
     source chatbot_env/bin/activate
     ```

4. **Install Dependencies**

   Ensure you have `pip` updated:

   ```bash
   pip install --upgrade pip
   ```

   Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. **Download NLTK Data**

   The script automatically downloads necessary NLTK data packages. However, if you encounter issues, manually download them:

   ```python
   import nltk
   nltk.download(['punkt', 'stopwords', 'wordnet'])
   ```

## Usage

### Initial Setup

1. **Prepare Training Data**

   The chatbot relies on text data to build its n-gram model. You can use books from Project Gutenberg as your training source.

2. **Running the Chatbot**

   Execute the main script:

   ```bash
   python chatbot.py
   ```

### Training the Model

Upon running the chatbot for the first time, if no `train.txt` file is found, the chatbot will prompt you to download books from Project Gutenberg.

1. **Download Books**

   - **Prompt:** "No training data found. Would you like to download from Project Gutenberg? (y/n)"
   - **Action:** Type `y` to proceed.

2. **Enter Book IDs**

   - **Prompt:** "Enter Project Gutenberg book ID (or 'done' to finish):"
   - **Action:** Enter the numeric ID of the desired book. For example, `1342` for "Pride and Prejudice."
   - **Finish:** Type `done` when you've added all desired books.

   **Note:** Ensure you have an active internet connection to download the books.

3. **Adding More Books (Optional)**

   After the initial training, you can choose to add more books to enhance the model.

   - **Prompt:** "Would you like to add more books? (y/n)"
   - **Action:** Type `y` to add more books following the same process.

### Starting the Chatbot

Once the model is trained, the chatbot enters the chat interface:

```bash
Chatbot is ready! Let's chat (type 'exit' to quit):
You: hi
Chatbot: Hmm... Hello! 😊
You: What is your name?
Chatbot: I think you know, but I'm just a chatbot! 😊
You: exit
Farewell! 🤔
```

**Commands:**

- **Exit:** Type `exit` to terminate the conversation.

## Project Structure

```
ngram-chatbot/
├── chatbot.py
├── requirements.txt
├── README.md
├── train.txt
└── ngram_model.db
```

- **chatbot.py:** Main script containing the chatbot logic.
- **requirements.txt:** Lists all third-party dependencies.
- **README.md:** Documentation file (this file).
- **train.txt:** Text file containing the training data (Project Gutenberg books).
- **ngram_model.db:** SQLite database storing n-gram counts.

## Configuration

### Parameters in `chatbot.py`

- **n:** The order of n-grams (default is 4, i.e., tetragrams). Adjusting this can impact the chatbot's context awareness.
- **top_k:** Vocabulary size limit (default is 15,000 words plus `<UNK>`). Determines which words are considered common enough to be included in the model.
- **max_words:** Maximum number of words in a generated response (default is between 15 to 25).
- **temperature:** Controls randomness in word selection (range typically between 0.6 to 0.9). Lower values make responses more deterministic.
- **max_repeats:** Limits the number of times a word or phrase can be repeated to prevent unnatural repetition.

*These parameters can be adjusted directly in the script to tweak the chatbot's behavior.*

## Contributing

Contributions are welcome! If you'd like to improve the chatbot or fix issues, follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/improve-chatbot
   ```

3. **Make Your Changes**

4. **Commit Your Changes**

   ```bash
   git commit -m "Improve response generation logic"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/improve-chatbot
   ```

6. **Create a Pull Request**

   Describe your changes and why they're beneficial.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **Natural Language Toolkit (nltk):** For providing essential NLP tools.
- **Project Gutenberg:** For offering a vast repository of free eBooks used as training data.
- **Contractions Library:** For simplifying the expansion of contractions in text.
- **Python Community:** For the wealth of resources and support.

## Contact

For any questions, suggestions, or feedback, feel free to reach out:

- **Email:** your.j.hn.w.nst.d@gmail.com
- **GitHub:** [@jhnwnstd](https://github.com/jhnwnstd)