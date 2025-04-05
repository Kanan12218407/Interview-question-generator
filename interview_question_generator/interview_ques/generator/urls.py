from django.urls import path
from .views import generate_question

urlpatterns = [
    path("", generate_question, name="generate_question"),
]
