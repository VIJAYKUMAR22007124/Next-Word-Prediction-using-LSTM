# Next-Word-Prediction-using-LSTM

## AIM

To develop an LSTM-based model for predicting the next word in a text corpus.

## Problem Statement 
This experiment aims to build a text generation model that predicts the next word in a sequence based on a given corpus. It involves preprocessing the text through tokenization and n-grams, followed by sequence padding. The model uses an embedding layer, bidirectional LSTM, and a dense layer with softmax for word prediction.
## Dataset
<img width="257" alt="{8AEB68D1-54D7-4568-B07B-A4E10A1B6733}" src="https://github.com/user-attachments/assets/9adada98-5e81-4a5e-b703-983d30070ea8">

## DESIGN STEPS

### Step 1:
Use fit_vectorizer to initialize and fit a TextVectorization layer on the corpus, converting words into integer tokens for further processing.

### Step 2:
Generate n-grams for each sentence using n_gram_seqs, creating sequential input data where each sequence includes multiple tokenized words.

### Step 3:
Pad the generated n-grams to a consistent length using pad_seqs, ensuring that all input sequences have the same size, suitable for model training.

### Step 4:
Split the padded sequences into features and labels, where the features are the sequence tokens excluding the last word, and the labels are the last word in each sequence.

### Step 5:
One-hot encode the labels using the total_words parameter to map each label to a vector, facilitating categorical prediction for word generation.

### Step 6:
Create a TensorFlow dataset with the features and one-hot encoded labels, and batch it for efficient training, using a batch size of 16.

### Step 7:
Build a sequential model with an Embedding layer for word representations, a Bidirectional LSTM layer for sequence learning, and a Dense layer with softmax activation for predicting the next word in the sequence.

### Step 8:
Compile and train the model using categorical cross-entropy loss and the Adam optimizer, monitoring accuracy during training to optimize the word prediction performance.








## PROGRAM
### Name: B VIJAY KUMAR 
### Register Number: 212222230173

### 1.fit_vectorizer function

```
def fit_vectorizer(corpus):
    """
    Instantiates the vectorizer class on the corpus

    Args:
        corpus (list): List with the sentences.

    Returns:
        (tf.keras.layers.TextVectorization): an instance of the TextVectorization class containing the word-index dictionary, adapted to the corpus sentences.
    """

    tf.keras.utils.set_random_seed(65) # Do not change this line or you may have different expected outputs throughout the assignment

    ### START CODE HERE ###

     # Define the object with appropriate parameters
    vectorizer = tf.keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation',  # Convert to lowercase and strip punctuation
        split='whitespace',  # Split on whitespace (default)
        ragged=True,  # Allow ragged tensors
        output_mode='int'  # Output as integers
    )
    
    # Adapt it to the corpus
    vectorizer.adapt(corpus)
    ### END CODE HERE ###

    return vectorizer
```
### 2. n_grams_seqs function
```
def n_gram_seqs(corpus, vectorizer):
    """
    Generates a list of n-gram sequences

    Args:
        corpus (list of string): lines of texts to generate n-grams for
        vectorizer (tf.keras.layers.TextVectorization): an instance of the TextVectorization class adapted in the corpus

    Returns:
        (list of tf.int64 tensors): the n-gram sequences for each line in the corpus
    """
    input_sequences = []

    ### START CODE HERE ###
    for sentence in corpus:
        # Vectorize the sentence to get the token indices
        vectorized_sentence =  vectorizer(sentence)

        # Generate n-grams for the vectorized sentence
        for i in range(2, vectorized_sentence.shape[0] + 1):  # Start from 2 to avoid the first token
            n_gram = vectorized_sentence[:i]
            input_sequences.append(n_gram)

    ### END CODE HERE ###

    return input_sequences
```

### 3. pad_seqs function
```
def pad_seqs(input_sequences, max_sequence_len):
    """
    Pads tokenized sequences to the same length

    Args:
        input_sequences (list of int): tokenized sequences to pad
        maxlen (int): maximum length of the token sequences

    Returns:
        (np.array of int32): tokenized sequences padded to the same length
    """

   ### START CODE HERE ###
    # Convert tensors to lists if necessary
    input_list = [seq if isinstance(seq, list) else seq.numpy().tolist() for seq in input_sequences]

    # Use pad_sequences to pad the sequences with left padding ('pre')
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        input_list,              # Use the list of lists for padding
        maxlen=max_sequence_len,  # Set the maximum length
        padding='pre',            # Pad sequences to the left (before the sequence)
        dtype='int32'             # Specify the output type as int32
    )
    ### END CODE HERE ###

    return padded_sequences
```

### 4. features_and_labels_dataset function
```
def features_and_labels_dataset(input_sequences, total_words):
    """
    Generates features and labels from n-grams and returns a tensorflow dataset

    Args:
        input_sequences (list of int): sequences to split features and labels from
        total_words (int): vocabulary size

    Returns:
        (tf.data.Dataset): Dataset with elements in the form (sentence, label)
    """
    ### START CODE HERE ###
    # Define the features by taking all tokens except the last one for each sequence
    features = [seq[:-1] for seq in input_sequences]

    # Define the labels by taking the last token for each sequence
    labels =  [seq[-1] for seq in input_sequences]

    # One-hot encode the labels using total_words as the number of classes
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)


    # Build the dataset using the features and one-hot encoded labels
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))


    # Batch the dataset with a batch size of 16
    batch_size = 16  # Feel free to adjust this based on the global variable, but should be <= 64
    batched_dataset = dataset.batch(batch_size)

    ### END CODE HERE ###

    return batched_dataset
```

### 5.create_model function
```
def create_model(total_words, max_sequence_len):
    """
    Creates a text generator model
    
    Args:
        total_words (int): size of the vocabulary for the Embedding layer input
        max_sequence_len (int): length of the input sequences
    
    Returns:
       (tf.keras Model): the text generator model
    """
    model = tf.keras.Sequential()

   ### START CODE HERE ###
    # Input layer shape is max_sequence_len - 1 because we removed the last word as a label
    model.add(tf.keras.layers.Input(shape=(max_sequence_len - 1,)))

    # Embedding layer
    model.add(tf.keras.layers.Embedding(input_dim=total_words, 
                                        output_dim=100, 
                                        input_length=max_sequence_len - 1))

    # Add a Bidirectional LSTM layer with 150 units
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))

    # Add a Dense layer with 'total_words' units and softmax activation
    model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    ### END CODE HERE ###

    return model
```




## OUTPUT
### 1. fit_vectorizer output

<img width="315" alt="{6C2DB403-D7DA-4496-AA1B-DD8C03567A65}" src="https://github.com/user-attachments/assets/1fe8f0ad-13b0-4af1-9c62-5a24b504037d">
<HR>
<img width="482" alt="{7F1F335B-BF96-42AB-A150-727DA314E6E5}" src="https://github.com/user-attachments/assets/1ea8891e-3c93-4c09-bf0a-be6b660fe2ab">


### 2. n_grams_seqs output
<img width="309" alt="{35CBB0D6-9738-4281-BC7A-21A278682CBE}" src="https://github.com/user-attachments/assets/3500bad2-a42f-499c-9046-19b8b1d38973">


### 3. pad_seqs output

<img width="303" alt="{EDCA699D-7B17-429E-998C-FFADE062841E}" src="https://github.com/user-attachments/assets/87e9fa67-1633-4ef8-86f2-7101f2dce25e">

### 4. features_and_labels_dataset output
<img width="332" alt="{0DDDA5DE-CF6F-43FB-BBEC-72FBA0F2226B}" src="https://github.com/user-attachments/assets/4e23b855-b855-4353-9d4d-fcb597a3820e">


### 5. Training Loss, Validation Loss Vs Iteration Plot

<img width="544" alt="{74CDAFFE-D801-46F9-845F-2699CC43A8B1}" src="https://github.com/user-attachments/assets/ef8ec94d-a40d-478b-93a5-10e1fc8ee060">


### 6. Sample Text Prediction
<img width="686" alt="{A5227BCA-20DA-441D-996C-600C57BA7CE9}" src="https://github.com/user-attachments/assets/8829ef02-c5b7-476e-8ee2-00aacbc5f03c">


## RESULT
Thus, a trained text generator model capable of predicting the next word in a sequence from the given corpus is successfully implelemted.
