from django.shortcuts import render
from .utils import classify_text  # Импортируем функцию классификации из utils.py

def index(request):
    result = None
    if request.method == 'POST':  # Если запрос POST, значит пользователь отправил текст для классификации
        text = request.POST.get('text')  # Получаем текст из формы
        if text:
            result = classify_text(text)  # Классифицируем текст
    
    return render(request, 'classify/index.html', {'result': result})  # Отправляем результат в шаблон
