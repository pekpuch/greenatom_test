from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Главная страница, где будет форма для классификации текста
]
