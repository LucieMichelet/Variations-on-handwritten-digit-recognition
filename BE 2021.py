# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 22:15:36 2021

@author: Lucie
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import biblio_BE as b


#%% importation des données
train_data = np.genfromtxt("mnist_train.csv", delimiter = "," )[1:6000,:]       # données d’entrainement
test_data = np.genfromtxt("mnist_test.csv", delimiter=  "," )[1:6000,:]         # données de test

"""
Nous avons diminué le nombre de données étudiées à cause d'un manque de capacité de nos appareils.
"""


#%%
# =============================================================================
#                          Partie 1 : Inverser le non-inversible
# =============================================================================
#%% P1 - Question 3

A0 ,y0 = b.apprentissage(train_data,0)                                          #apprentissage au nombre 0

C = b.iterationQR(A0.T@A0)                                                      #valeurs propres de (A.T)A
rang = np.linalg.matrix_rank(A0)                                                #rang approximé de A

"""
la matrice A.T A n'est pas inversible car certaines de ses valeurs propres sont nulles.
"""

#%% P1 - Question 4

U,S,Vt = np.linalg.svd(A0)
rangsvd = b.rangsvd(A0)                                                         #rang approximé de A avec la décomposition SVD

"""
Le rang calculé avec la décomposition svd est plus grand (785) qu'avec la méthode numpy (656).
On en conclue que la méthode svd est plus précise.
"""

#%% P1 - Question 5 - a) b) c) création des matrices A et y pour chaque chiffre

listA = []
listy = []

for nb in range(0,10):
    
    a, Y = b.apprentissage(train_data, nb)
    listA.append(a)
    listy.append(Y)   
    
listA = np.array(listA)
listy = np.array(listy)

#%% P1 - Question 5

for A in listA : 
    print(b.defsympos(A,10))
    

"""
La matrice est bien symetrique et définie positive quelque soit la valeure de epsilon et du chiffre étudié.
"""

#%%

def resChol(nombredetection,epsilon):
    A = listA[nombredetection]
    y = listy[nombredetection]
    AE = A.T@A + epsilon*np.eye(785)
    B = b.cholesky(AE)
    X = b.descente(B,A.T@y)
    sol = b.monter(B.T,X)
    Nvp=0
    Nfn=0
    Nfp=0
    Nvn=0
    labels_test = test_data[:,0]
    N = len(labels_test)
    
    for k in range(0,N) :
        
          T = b.f(test_data[k,1:],sol)    
          
          if T>0 and labels_test[k]==nombredetection :
             Nvp=Nvp+1
             
          if T<=0 and labels_test[k]==nombredetection :
             Nfn=Nfn+1
             
          if T>0 and labels_test[k]!=nombredetection :
             Nfp=Nfp+1
             
          if T<=0 and labels_test[k]!=nombredetection :   
             Nvn=Nvn+1
             
    txreussite=(Nvp+Nvn)/N
    Mc=np.array([[Nvp, Nfn],[Nfp, Nvn]])   
    
    return sol, txreussite, Mc


#%% affichage des données 

for i in range (10):

    sol, T, Mc = resChol(i,1)
    print('Matrice de confusion pour %s: ' %i +'\n', Mc)
    print('Taux de réussite : ', T)
    

#%% P1 - Question 5 - d)
taux = []
x = np.linspace(1,10**2)

for i in x:
    sol, t, Mc = resChol(0,i)
    taux.append(t)

plt.figure('Taux de réussite')
plt.xlabel('espilon')
plt.ylabel('Taux de réussite')
plt.plot(x,taux, c='b')
   
plt.show()

emax = x[taux.index(max(taux))]

"""
On en conclut qu'il faut réduire l'étude en découpant les intervalles à cause du manque de précision.
On trouve que le meilleur epsilon pour le chiffre 0 vaut environ 333 265.
"""

#%% P1 - Question 5 - e) 

def optimus_e():
    
    N = 50                                                                      #nb de point par intervalle
    x = [np.linspace(10**(-10),10**(-8),N),np.linspace(10**(-8),10**(-6),N),np.linspace(10**(-6),10**(-4),N),np.linspace(10**(-4),10**(-2),N),np.linspace(10**(-2),1,N),np.linspace(1,10**2,N),np.linspace(10**2,10**4,N),np.linspace(10**4,10**6,N),np.linspace(10**6,10**9,N)]        # Création des 9 intervalles d'étude
    F = open("Solutions",'w')                                                   #création d'un fichier de sauvegarde des données                         
    
    for j in range(0,10):
        
        F.write("\n=================================\nEtude du chiffre {}\n=================================\n".format(j))

        solmax = []
        tauxmax =[]
        R = []

        for k in range(len(x)):                                                 #boucle sur chaque intervalle d'étude, k : la k-ième intervalle
            taux = []
            matrice = []
            solution = []
                   
            for i in x[k]:
                sol, t, Mc = resChol(j,i)                                       #j : le nb étudié, i : epsilon
                taux.append(t)                                                  #récupération des données calculées dans des listes
                matrice.append(Mc)
                solution.append(sol)
            
            
            plt.plot(x[k],taux, c='b')                                          #création d'un graphique représentant le taux par intervalle pour avoir une vision globale
            plt.title('Taux de réussite du chiffre {}'.format(j))
            plt.xlabel('espilon')
            plt.ylabel('Taux de réussite')
            plt.savefig("Taux de réussite du chiffre {} pour l'intervale {}".format(j,k))   # enregistrement du graphique dans le dossier
            plt.clf()

            b = taux.index(max(taux))                                           #récupération de la meilleure valeur par intervalle
            tauxmax.append(max(taux))
            emax = x[k][b]
            Mcmax = matrice[b]
            solmax.append(solution[b])
                    
            
            F.write("Intervale numéro {}\n".format(k))
            F.write("Taux de réussite max : {}\nespilon max : {}\nMatrice de confusion associée : {}\n".format(max(taux),emax,Mcmax))
        
        B = tauxmax.index(max(tauxmax))                                         #récupération de la meilleure valeur pour les 9 intervalles pour avoir les solutions
        R = np.array(solmax[B])
        np.save("solution {}".format(j),R)
            
    F.close()
    
#optimus_e()                                                                     #lancement de la fonction

#%% P1 - Question 6 - création de la matrice solution des chiffres de 0 à 9 


sol0 = np.load("solution 0.npy").reshape(785,1)
sol1 = np.load("solution 1.npy").reshape(785,1)
sol2 = np.load("solution 2.npy").reshape(785,1)
sol3 = np.load("solution 3.npy").reshape(785,1)
sol4 = np.load("solution 4.npy").reshape(785,1)
sol5 = np.load("solution 5.npy").reshape(785,1)
sol6 = np.load("solution 6.npy").reshape(785,1)
sol7 = np.load("solution 7.npy").reshape(785,1)
sol8 = np.load("solution 8.npy").reshape(785,1)
sol9 = np.load("solution 9.npy").reshape(785,1)

SOL = np.concatenate((sol0, sol1, sol2, sol3, sol4, sol5, sol6, sol7, sol8, sol9), axis= 1)


#%% fonction reconnaissance des chiffres manuscrits 

def fglobale(x,SOL):
    
    L = []
    for i in range (0,10):
        w = SOL[:784 , i]
        b = SOL [-1 , i]
        f = w@x + b
        L.append(f)
        
    return L, L.index(max(L))  

    
i = np.random.randint(0,6000)
x = test_data[i,1:].reshape(784,1)
resulatC = fglobale(x,SOL)[1]  

#%%
# =============================================================================
#                            Partie 2 : Procuste 
# =============================================================================

#test d'affichage du 0 moyen 

nombredetection = 0                                                             #ici c'est le nombre qu'on veut détecter : 0
valeurs = train_data[:,0]                                                       #récupère la première colonne de train_data, ce sont les indices qui permettent de savoir de quel nombre il s'agit
indiceu = np.where(valeurs == nombredetection)                                  #récupère touts les indices qui sont égaux au nombre qu'on veut détecter 
u = train_data[:,1 :][indiceu]                                                  #récupère les images vectorisées de tous les 0 dans une matrice de taille (nombre de 0, taille de l'image)
l,c = u.shape                                                                   #l = nombre de ligne et c = nombre de colonnes de la matrice d'images de 0
L = []         
                                                                                #création d'une liste vide dans laquelle on va ajouter les valeurs moyennes de chaque pixels de l'image
for i in range(0,c):   
                                                                                #boucle permettant de parcourir les colonnes de notre matrice d'images de 0 u
    moy = np.mean(u[:,i])                                                       #calcul de la moyenne de tous les termes d'une colonne
    L.append(moy)              
                                                                                #on ajoute cette moyenne de colonne dans la liste
zeromoyen = np.array(L).reshape((1,c))  
                                                                                #transformation de la liste en vecteur de taille (1,784)
#%% Affichage du Zéro moyen 

img=zeromoyen.reshape((28,28))                                                  #transformation du vecteur image en image de 28 par 28 pixels
cv2.imshow('chiffre', np.uint8(img)) 
                                                                                #affichage de l'image sans oublier de la convertir en uint8
#%% Généralisation 

CM = b.chiffremoyen(train_data) 
                                                                                #test sur notre DataBase d'entrainement
#%% Affichage 

for i in range(0,10):     
                                                                                #boucle parcourant tous les chiffres de 0 à 9
    image = CM[i].reshape((28,28))                                              #on selectionne chaque ligne de la matrice en les convertissant en matrice 28 par 28 
    cv2.imshow('chiffre moyen {}'.format(i), np.uint8(image))                   #affichage de l'image 28 par 28 pixels en la convertissant en uint8

#%% approche de procuste 

i = np.random.randint(0,6000)
xp = test_data[i,1:].reshape((784,1))
veccomp,resultatp = b.comparaison(x, CM)

#%% Taux de reconnaissance 

Tr = b.TauxReconnaissanceProcuste(test_data,CM)



#%%
# =============================================================================
#                          Partie 3 : Le mélange 
# =============================================================================
#%% Finalisation 

def Reconnaissance(i, coeff):

    v = coeff * (1/b.comparaison(test_data[i,1:][np.newaxis].T, CM)[0])   
    rep = np.array(fglobale(test_data[i,1 :].reshape((784,1)),SOL)[0]).reshape(np.shape(v))
    resultat = v + rep
    s = np.max(resultat)
    ind = int(np.where(resultat==s)[0][0])

    return resultat, ind

#%% calcul du taux de réussite 

def TauxGlobal(coeff):

    valeur = test_data[:,0]
    l = len(valeur)
    Nv = 0

    for i in range(0, l):
        #x = test_data[i,1:].reshape((784,1))
        resultat, chiffre = Reconnaissance(i,coeff)
        if chiffre == valeur[i]: 
            Nv+=1
    print(Nv)

    return Nv/l


#%% test avec un coefficient quelconque 

i = 200
x = test_data[i,1:].reshape(28,28)
cv2.imshow("Chiffre",x)
resultatf, ind = Reconnaissance(i, 240)
print(ind)


#%% Taux final avec un coefficient quelconque

Tauxf = TauxGlobal(240) 

#%% détermination du meilleur coeff 

Tauxlist = []
cof = np.linspace(10**(-9), 10, 17)

for coeff in cof:
    
    val = TauxGlobal(coeff)
    Tauxlist.append(val)
    
maxcoeff = Tauxlist.index(max(Tauxlist))
coeffFinal = cof[maxcoeff]


#%%affichage des données 

plt.figure()
plt.plot(cof, Tauxlist,'-r')
plt.savefig("Taux de réussite en fonction du coefficient")
plt.show()