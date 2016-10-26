#!usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Se construye funcion score_model, la cual evalua el desempeno de un determinado clasificador
def score_model(model, x, y, xt, yt, text):
    acc_train = model.score(x,y)
    acc_test = model.score(xt, yt)
    print 'Precisión datos de entrenamiento %s: %f'%(text, acc_train)
    print 'Precisión datos de prueba %s: %f'%(text, acc_test)
    print 'Análisis detallado de resultados sobre set de prueba:'
    print (classification_report(yt, model.predict(xt), target_names = ['clase 1', 'clase 0']))

#Implementacion clasificador bayesiano ingenuo binario
def NAIVE_BAYES(x, y, xt, yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    score_model(model, x, y, xt, yt, 'BernoulliNB')
    return model

#Implementacion clasificador bayesiano ingenuo multinomial
def MULTINOMIAL(x,y,xt,yt):
    model = MultinomialNB()
    model = model.fit(x,y)
    score_model(model, x, y, xt, yt, "MULTINOMIAL")
    return model

#Implementacion modelo de regresion logistica regularizado
def LOGIT(x,y,xt,yt, bestvalue):
    Cs = [0.01, 0.1, 10, 100, 1000]
    if bestvalue == 0:
        for C in Cs:
            print "Usando C= %f"%C
            model = LogisticRegression(penalty='l2', C=C)
            model= model.fit(x,y)
            score_model(model, x, y, xt, yt, "LOGISTIC")
    else:
        model = LogisticRegression(penalty='l2', C=bestvalue)
        model = model.fit(x,y)
        score_model(model,x,y,xt,yt, "LOGISTIC")
        return model

#Implementacion de modelo SVM lineal
def SVM(x,y,xt,yt, bestvalue):
    Cs = [0.01, 0.1, 10, 100, 1000]
    if bestvalue == 0: #Cuando el valor de C aun no ha sido escogido, se prueban todos los C
        for C in Cs:
            print "El valor de C que se esta probando: %f"%C
            model = SVC(C=C, kernel='linear', probability=True)
            model = model.fit(x,y)
            score_model(model, x, y, xt, yt, "SVM")
    else: #Se entrena/ajusta modelo con el valor de C escogido
        model = SVC(C=bestvalue, kernel='linear', probability=True)
        model = model.fit(x,y)
        score_model(model, x, y, xt, yt, "SVM")
        return model