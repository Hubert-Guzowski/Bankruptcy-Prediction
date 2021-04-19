from django.shortcuts import render


def homepage(request):
    return render(request, 'interface/homepage.html')
