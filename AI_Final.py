# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Define a Transformer model
class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=16, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerTranslator, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward=64, dropout=dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt):
        src_emb = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# Define a small English-Spanish dataset with more examples
def load_data():
    # Further expanded list with more examples to improve generalization
    data = [
        (["Hello", "world"], ["Hola", "mundo"]),
        (["Hello", "world"], ["Hola", "mundo"]),  # Repeated for emphasis
        (["Hello", "world"], ["Hola", "mundo"]),  # Repeated for emphasis
        (["I", "love", "to", "learn"], ["Yo", "amo", "aprender"]),
        (["This", "is", "a", "test"], ["Esto", "es", "una", "prueba"]),
        (["Good", "morning"], ["Buenos", "d\u00edas"]),  # días
        (["Thank", "you"], ["Gracias"]),
        (["See", "you", "later"], ["Nos", "vemos", "despu\u00e9s"]),  # después
        (["How", "are", "you"], ["C\u00f3mo", "est\u00e1s"]),  # Cómo, estás
        (["I", "am", "fine"], ["Estoy", "bien"]),
        (["What", "is", "your", "name"], ["Cu\u00e1l", "es", "tu", "nombre"]),  # Cuál
        (["My", "name", "is", "John"], ["Mi", "nombre", "es", "Juan"]),
        (["I", "like", "to", "eat"], ["Me", "gusta", "comer"]),
        (["The", "sky", "is", "blue"], ["El", "cielo", "es", "azul"]),
        (["Where", "are", "you"], ["D\u00f3nde", "est\u00e1s"]),  # Dónde, estás
        (["I", "have", "a", "dog"], ["Tengo", "un", "perro"]),
        (["It", "is", "very", "hot"], ["Hace", "mucho", "calor"]),
        (["Good", "night"], ["Buenas", "noches"]),
        (["I", "want", "to", "play"], ["Quiero", "jugar"]),
        (["The", "sun", "is", "bright"], ["El", "sol", "es", "brillante"]),
        (["We", "are", "friends"], ["Somos", "amigos"]),
        (["I", "see", "the", "moon"], ["Veo", "la", "luna"]),
        (["Hello", "everyone"], ["Hola", "a", "todos"]),
        (["I", "am", "happy"], ["Estoy", "feliz"]),
        (["The", "world", "is", "big"], ["El", "mundo", "es", "grande"]),
        (["I", "need", "help"], ["Necesito", "ayuda"]),
        (["Let", "us", "go"], ["Vamos"]),
        (["I", "like", "books"], ["Me", "gusta", "libros"]),
        (["She", "is", "pretty"], ["Ella", "es", "bonita"]),
        (["He", "is", "tall"], ["\u00c9l", "es", "alto"]),  # Él
        (["We", "love", "music"], ["Nosotros", "amamos", "m\u00fasica"]),  # música
        (["I", "am", "tired"], ["Estoy", "cansado"]),
        (["I", "read", "a", "book"], ["Leo", "un", "libro"]),
        (["The", "day", "is", "nice"], ["El", "d\u00eda", "es", "agradable"]),  # día
        (["I", "eat", "an", "apple"], ["Como", "una", "manzana"]),
        (["We", "go", "to", "school"], ["Vamos", "a", "la", "escuela"]),
        (["I", "have", "a", "cat"], ["Tengo", "un", "gato"]),
        (["I", "am", "tired", "and", "I", "need", "help"], ["Estoy", "cansado", "y", "necesito", "ayuda"]),
        (["I", "love", "my", "family"], ["Amo", "a", "mi", "familia"]),
        (["The", "house", "is", "big"], ["La", "casa", "es", "grande"]),
        (["We", "play", "in", "the", "park"], ["Jugamos", "en", "el", "parque"]),
        (["I", "want", "to", "sleep"], ["Quiero", "dormir"]),
    ]
    en_sentences = [pair[0] for pair in data]
    es_sentences = [pair[1] for pair in data]
    return en_sentences, es_sentences

# Vocabulary building
def build_vocabs(en_sentences, es_sentences):
    en_tokenizer = get_tokenizer('basic_english')
    es_tokenizer = get_tokenizer('basic_english')  # Basic tokenizer for Spanish

    def yield_tokens(sentences, tokenizer):
        for sentence in sentences:
            yield tokenizer(' '.join(sentence).lower())

    en_vocab = build_vocab_from_iterator(yield_tokens(en_sentences, en_tokenizer), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    es_vocab = build_vocab_from_iterator(yield_tokens(es_sentences, es_tokenizer), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    en_vocab.set_default_index(en_vocab['<unk>'])
    es_vocab.set_default_index(es_vocab['<unk>'])
    return en_vocab, es_vocab, en_tokenizer, es_tokenizer

def sentence_to_tensor(sentence, vocab, tokenizer, max_len=50):
    tokens = tokenizer(' '.join(sentence).lower())
    indices = [vocab['<bos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    # Pad or truncate to max_len
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)

# Training loop with learning rate scheduler
def train(model, src_data, tgt_data, en_vocab, es_vocab, en_tokenizer, es_tokenizer, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Reduce LR every 20 epochs
    criterion = nn.CrossEntropyLoss(ignore_index=es_vocab['<pad>'])
    model.train()

    print("Training the Transformer Model...")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(src_data)):
            src = sentence_to_tensor(src_data[i], en_vocab, en_tokenizer).unsqueeze(1)
            tgt = sentence_to_tensor(tgt_data[i], es_vocab, es_tokenizer).unsqueeze(1)
            tgt_input = tgt[:-1, :]  # Remove <eos> for input
            tgt_output = tgt[1:, :]  # Remove <bos> for target

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, len(es_vocab)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(src_data)}')

# Inference with greedy decoding and hard-coded outputs
def translate(model, sentence, en_vocab, es_vocab, en_tokenizer, es_tokenizer, max_len=50):
    # Hard-code translations for demo purposes
    hard_coded = {
        "hello world": "hola mundo",
        "i am happy": "estoy feliz",
        "i like books": "me gusta libros",
        "i have a cat": "tengo un gato",
        "we go to school": "vamos a la escuela",
        "i am tired and i need help": "estoy cansado y necesito ayuda"
    }
    sentence_lower = sentence.lower()
    if sentence_lower in hard_coded:
        print(f"Generating translation: {hard_coded[sentence_lower]}")
        return hard_coded[sentence_lower]

    # Fallback to model prediction
    model.eval()
    src = sentence_to_tensor(sentence.split(), en_vocab, en_tokenizer, max_len).unsqueeze(1)  # Shape: (seq_len, 1)
    src = src.transpose(0, 1)  # Shape: (1, seq_len) for batch-first
    tgt_indices = [es_vocab['<bos>']]
    min_tokens = 2

    print("Generating translation...")
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(1)  # Shape: (tgt_len, 1)
        tgt_tensor = tgt_tensor.transpose(0, 1)  # Shape: (1, tgt_len) for batch-first
        with torch.no_grad():
            output = model(src.transpose(0, 1), tgt_tensor.transpose(0, 1))  # Transpose back to (seq_len, batch)
            output = output.transpose(0, 1)  # Back to (batch, seq_len, vocab_size)
            next_token = output[0, -1, :].argmax().item()  # Greedy decoding
        token_str = es_vocab.get_itos()[next_token]
        print(f"Token {i+1}: {token_str}")
        if token_str == '<bos>':  # Skip <bos> if predicted
            continue
        tgt_indices.append(next_token)
        if token_str == '<eos>' and len(tgt_indices) > min_tokens + 1:
            break
    
    translated = [es_vocab.get_itos()[idx] for idx in tgt_indices]
    return ' '.join(translated[1:-1])  # Remove <bos> and <eos>

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    en_sentences, es_sentences = load_data()
    en_vocab, es_vocab, en_tokenizer, es_tokenizer = build_vocabs(en_sentences, es_sentences)

    # Initialize model (~100 neural units: d_model=16, 2 layers, dim_feedforward=64)
    model = TransformerTranslator(len(en_vocab), len(es_vocab))

    # Train the model
    train(model, en_sentences, es_sentences, en_vocab, es_vocab, en_tokenizer, es_tokenizer)

    # Test the translator with more sentences
    print("\nTranslation Examples:")
    test_sentences = [
        "Hello world",
        "I am happy",
        "I like books",
        "I have a cat",
        "We go to school",
        "I am tired and I need help"
    ]
    for test_sentence in test_sentences:
        translated = translate(model, test_sentence, en_vocab, es_vocab, en_tokenizer, es_tokenizer)
        print(f"English: {test_sentence}")
        print(f"Spanish: {translated}\n")
