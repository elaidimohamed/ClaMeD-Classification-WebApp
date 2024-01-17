from django.urls import path
from . import views


urlpatterns = [
    path('',views.main),
    path('home',views.home),
    path('user', views.predict, name='predict')
]

