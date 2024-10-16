from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
import torch

# Инициализация токенайзера и модели

tokenizer = ElectraTokenizer.from_pretrained('data/tokinizer')

model = ElectraForSequenceClassification.from_pretrained('data/tokinizer', num_labels=8)  
model.load_state_dict(torch.load('data/frozen2.pth', weights_only=True))

model = model.eval()

# Установка режима оценки (eval)
model.eval()

def classify_text(text):
    # Токенизация входного текста
    classes = [1, 2, 3, 4, 7, 8, 9, 10]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Прогон текста через модель
    with torch.no_grad():
        outputs = model(**inputs)

    # Получение логитов (сырых предсказаний) и выбор наибольшего
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return classes[predicted_class]  # Возвращаем предсказанный класс
