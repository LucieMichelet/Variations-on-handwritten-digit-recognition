# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:50:57 2021

@author: Lucie
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import seaborn as sns



#%% Procuste 


def apprentissage(train_data, nombredetection):
    
    """
    param train_data :
    param nombredetection : 
    returns A :
    returns y : 
    """
    
    valeurs = train_data[:,0]
    indiceu = np.where(valeurs == nombredetection)
    indicev = np.where(valeurs != nombredetection)
    u=train_data[:,1:][indiceu]
    v=train_data[:,1:][indicev]
    A=np.block([[u, np.ones((len(indiceu[0]),1))], [v ,np.ones((len(indicev[0]),1)) ]])
    y=np.block([[np.ones((len(indiceu[0]),1))], [-np.ones((len(indicev[0]),1))]])
    
    return A, y      



def iterationQR(A) :
    
    """
    param A : la matrice a transformer
    return diagH :
    """
    
    Q,R=np.linalg.qr(A)    
    for i in range(0,50) :
        H=Q@R
        Q,R=np.linalg.qr(H)
        
    diagH = np.diagonal(H)
        
    return diagH



def rangsvd(A) :
    
    """
    param A : matrice
    return r : rang de la matrice
    """
    
    r=0
    U,S,Vt = np.linalg.svd(A)
    for i in range(len(S)):
        if S[i] != 0:
            r += 1
    return r
        
 
    
    
def defsympos(A,k):
    
    """
    param A : La matrice testée
    param k : valeur de epsilon max à tester
    return: True si la matrice est inversible, False si non
    """   

    e=1
    condition = True 
    
    while condition == True and e<=k :
        M = A.T@A + e*np.eye(785)
        vpM = iterationQR(M) 
        
        for i in range(785):
            if vpM[i] > 0 and (M.T == M).all() :
                condition = True
            else :
                condition = False
                break   
        e +=1
        
    return condition




def cholesky(A):

    """
    :param A: matrice inversible symétrique definie positive (n, n)
    :return C: matrice triangulaire inférieur (n, n) tel que A = CC.T
    """

    n = A.shape[0]
    C = np.eye(n)

    C[0, 0] = np.sqrt(A[0, 0])

    for i in range(1, n):
        C[i, 0] = A[i, 0] / C[0, 0]

    for i in range(1, n):
        C[i, i] = np.sqrt(A[i, i] - np.sum(C[i, :i]**2))

        for j in range(i + 1, n):
            C[j, i] = (A[j, i] - np.sum(C[j, :i] * C[i, :i])) / C[i, i]

    return C




def descente(L, b):

    """
    :param L: matrice triangulaire inferieur (n, n)
    :param b: vecteur colonne (n, 1)
    :return: vecteur colonne y (n, 1) solution de l'équation Ly = b
    """

    n = L.shape[0]
    y = np.zeros((n, 1))


    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = b[i]

        for j in range(i):
            y[i] -= L[i, j] * y[j]

        y[i] /= L[i, i]

    return y

 
 

def monter(U, y):

    """
    :param U: matrice triangulaire supérieur (n, n)
    :param y: vecteur colonne (n, 1)
    :return: vecteur colonne x (n, 1) solution de l'équation Ux = y
    """

    n = U.shape[0]
    x = np.zeros((n, 1))
 
    x[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = y[i]

        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]

        x[i] /= U[i, i]

    return x

 

def f(x,sol) : 
    
    """
    param x :
    param sol :
    return = 
    """
    
    w=sol[:-1,0].reshape((784,1))
    b=sol[-1,0]
    
    return w.T @ x + b



def chiffremoyen(train_data):
    
    """ 
    Cette fonction permet de créer une matrices où chaque lignes correspondent 
    à l'image moyenne d'un chiffre de 0 à 9 à partir d'une base de données de 
    chiffres manuscrit
    """
    L = []      
                                                                #initialisation de la liste qui va comporter la moyenne de chaque pixels pour chaque chiffre
    for nb in range(0,10):      
                                                                                #boucle permettant de parcourir les chiffres de 0 à 9 
        nombredetection = nb
        valeurs = train_data[:,0]
        indiceu = np.where(valeurs == nombredetection)
        u = train_data[:,1 :][indiceu]
        l,c = u.shape
        
        for i in range(0,c): 
            moy = np.mean(u[:,i])
            L.append(moy)
    chiffremoy = np.array(L).reshape((10,c))  
                                                                                #transformation de la liste en matrice (10,784) où chaque ligne correspond au chiffre moyen de 0 à 9 
    return chiffremoy



def procuste(A,B) :
    
    """
    Cette fonction permet de calculer le rapport d'homotécie, une transformation 
    orthogonale ainsi qu'un vecteur de translation de la transformation affine 
    de l'analyse procustéenne. Elle renvoie également l'erreur de transformation.
    """
    m,n=np.shape(B)
    aG=np.zeros((m,1))
    
    for i in range(0,n) :
          aG=aG+(1./n)*A[:,i][np.newaxis].T
    bG=np.zeros((m,1))
    
    for i in range(n) :
          bG=bG+(1./n)*B[:,i][np.newaxis].T
          
    u=np.ones((1,n))
    Ag=A-np.dot(aG,u)
    Bg=B-np.dot(bG,u)
    P=np.dot(Ag,Bg.T)
    Ug,Sg,Vg=np.linalg.svd(P)
    X=np.dot(Vg.T,Ug.T)
    l=np.trace(np.diag(Sg))/np.linalg.norm(Ag,'fro')**2
    t=bG-l*np.dot(X,aG)
    
    return l,X,t,np.linalg.norm((B-l*X@A-t@u)**2)



def comparaison(x,CM): 
    
    """
    Cette fonction permet de comparer une image vectorisée du train_test avec
    celles des chiffres moyens. A l'aide de la fonction procuste on évalue l'erreur 
    de transformation entre les images puis renvoie l'indice du chiffre dont
    l'erreur est la plus faible ainsi que un vecteur contenant les erreurs pour 
    chaque chiffre moyen. 
    """
    B = x.T
    L = []
    
    for i in range(0,10):
        A = CM[i,:].reshape(1,784)
        v = procuste(A,B)[3]
        L.append(v)
        
    veccomp = np.array(L)
    resultat = int(np.where(veccomp == np.min(veccomp))[0][0])
    
    return veccomp, resultat



def TauxReconnaissanceProcuste(test_data,CM):
    
    """
    Cette fonction permet de calculer le taux de réussite en appliquant l'analyse 
    procustéenne
    
    """
    valeur = test_data[:,0]
    l = len(valeur)
    Nv = 0
    
    for i in range(0, l):
        x = test_data[i,1:].reshape((784,1))
        veccomp, resultat = comparaison(x,CM)
        
        if resultat == valeur[i]: 
            Nv+=1
            
    Tr = Nv/l
    
    return Tr


#%% ACP


def Xmi(col) :
    """
    Il s'agit de l'Espérance : la somme des termes divisé par le nombre de termes

    Parameters
    ----------
    col : TYPE np.array
        DESCRIPTION. un array en forme de colonne

    Returns
    -------
    TYPE float
        DESCRIPTION. C'est l'Espérance

    """
    return np.sum(col)/np.shape(col)[0]

def Smi(col) :
    """
    Il s'agit de la Variance

    Parameters
    ----------
   col : TYPE np.array
       DESCRIPTION. un array en forme de colonne

   Returns
   -------
   TYPE float
       DESCRIPTION. C'est la Variance

    """
    return 1/(np.shape(col)[0]-1)*np.sum((col-Xmi(col))**2)

def sigma(col) :
    """
   Il s'agit de l'écart-type

   Parameters
   ----------
  col : TYPE np.array
      DESCRIPTION. un array en forme de colonne

  Returns
  -------
  TYPE float
      DESCRIPTION. C'est l'écart-type
      
    """
    return Smi(col)**0.5

def centre_red(R) :
    """
    Permet de créer la matrice centrée-réduite à partir d'un tableau de données'

    Parameters
    ----------
    R : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier

    Returns
    -------
    Rcr : TYPE np.array
        DESCRIPTION. Tableau de la même taille que R avec des valeurs centrée-réduites

    """
    Rcr = np.zeros(np.shape(R)) 
                                                #On créer la matrice centrée réduite Rcr qu'on va remplir
    for i in range(np.shape(R)[1]) :            #On parcourt la matrice R colonne par colonne
        col = R[:,i]                            #On associe la i-ème colonne de R à la variable col
        col_cr = (col-Xmi(col))/(sigma(col)*(np.shape(col)[0]-1)**0.5)  #On calcule la colonne centrée-réduite à partir de col
        Rcr[:,i] = col_cr                       #On range notre variable dans la matrice créée au début de notre fonction
    
    return Rcr

def ACP(X,q) :
    """
    Permet d'afficher la matrice des coefficients de corrélation

    Parameters
    ----------
    X : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier
    q : TYPE, optional int
        DESCRIPTION. The default is 0. Le nombre de directions principales à prendre en compte car elle contiennent de l'information

    """
    
    Xcr = centre_red(X)
    U,S,Vt = np.linalg.svd(Xcr)
    compt = 0
    d = Xcr.shape[1]
    L = []
    
    for i in S :
        L.append(i)
        
        if i >= sum(L)/d :
            compt+=1
        q = compt

    U,S,Vt = np.linalg.svd(Xcr)
    Xq = Xcr@(U[:d,:q])
    
    return Xq, q


def rootW(taille) :
    W = np.zeros((taille,taille))
    for i in range(taille) :
        for j in range(taille) :
            W[i,j] = np.random.randint(0,255)**0.5
    return W

#%% ACP pondéré

def CovPond(X,W) :
    """
    Renvoie la matrice de corrélation pondérée
    Parameters
    ----------
    X : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier
    W : TYPE np.array
        DESCRIPTION  matrice diagonale avec les valeurs des pixels
    
    """
    Xcr = centre_red(X)
    Cov = 1/np.sum(W**2)*W@Xcr.T@Xcr@W
    return Cov 

def ACPpond(X,q,W) :
    """
    Permet d'afficher la matrice des coefficients de corrélation

    Parameters
    ----------
    X : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier
    q : TYPE, optional int
        DESCRIPTION. The default is 0. Le nombre de directions principales à prendre en compte car elle contiennent de l'information

    """

    Xcr = centre_red(X)
    U,S,Vt = np.linalg.svd(Xcr)
    compt = 0
    d = Xcr.shape[1]
    L = []
    
    for i in S :
        L.append(i)
        
        if i >= sum(L)/d :
            compt+=1
            
    q = compt
    Xcr = centre_red(X)
    U,S,Vt = np.linalg.svd(Xcr)
    CovX = CovPond(X, W)
    Xq = Xcr@(U[:d,:q])
    
    return Xq, CovX, q
    

#%% K moyen

def barycentre(S):
    # retourne un matrice de taille (k, p) ou chaque ligne est le barycentre de tous les points d'un catégorie S[k]
    return np.array([np.sum(s, axis=0, dtype=float) / len(s) for s in S])


def norme(x):
    # norme 2 de x
    return np.linalg.norm(x)


def Kmoy2(A, k, err=0.01, show=False, save=False):
    """
    L'algorithme des k-means permette de catégoriser / classifier les objets d'un EV en fonction de leurs distances
    euclidiennes relative. Dans notre cas il permet de séparer en k catégorie un nuage de point que l'on
    sait 'relativement ordonné' grace a un travail de décomposition en vecteur propre ACP effectué précédements.
    :param A: la matrice de taille (m, p) que souhaite classifier
    :param k: le nombre de groupe / catégorie que l'on souhaite créer
    :param err: condition d'arrete de l'algorithme : c'est la variation minimal entre les barycentres muu entre 2
                ittérations pour stopper l'algorithme
    :param show: if True affiche en temps réel l'évolution de l'algorithme ittération par ittération
    :param save: if True enregistre le résultat que l'on affiche sur un plan (fonction pour p = 2)
    :return S: la meilleur partition des lignes de A,
               une liste de longueur k qui contient les lignes de A trier selon k catégories exemple,
               pour k = 3: S = [[.., .., ..], [.., .., .., .., ..], [.., ..]]
    """

    # on récupère les dimensions de A
    m, p = A.shape
    # on stocke quelque couleur pour le future affichage
    color = ['b', 'g', 'k', 'y', 'm', 'pink']

    # ------------------------------------- initialistation de l'algorithme --------------------------------------
    # on choisie k ligne différente de A au hasard
    # on recommence tant que l'on a pas k indice différent
    indice = np.random.choice(m, k)
    while (np.unique(indice) != indice).all():
        indice = np.random.choice(m, k)

    # muu est un matrice de taille (k, p) qui contient les coordonnées des barycentres des catégories que l'on cherche
    # on initialise ces barycentres de manière aléatoire a partir des indices
    muu = A[indice, :]
    # on fixe delta_muu est notre variable de souvenir (ittération n-1)
    delta_muu = muu

    # ------------------------------------------- algorithme k-means -------------------------------------------
    # tant que l'algorithme n'est pas stable, que les barycentres se deplacent encore on ittere l'algo
    while norme(delta_muu) > err:
        # on initialise notre partition des lignes de A en k catégorie : [[], [], .., []]
        S = [[] for l in range(k)]

        if show:
            plt.clf()

        # pour chaque ai les ligne de A, on clacule sa distance aux k-barycentres et on ajoute ai au nuage de point
        # dont le barycentre est le plus proche
        for i in range(m):
            ai = A[i, :]

            # on calcule les distances entre ai et les k-barycentres avec la norme 2
            D = [norme(ai - muu[j, :]) for j in range(k)]
            # on récupère l'indice du nuage le plus proche de ai, avec j la catégorie qui lui conviendrait le mieux
            j, _ = min(enumerate(D), key=lambda x: x[1])
            # on ajoute la ligne ai a la meilleur catégorie au vu des barycentres de cette ittération
            S[j].append(ai)

            if show:
                plt.scatter(ai[0], ai[1], c=color[j])

        # il se peut qu'un catégorie de S soit vide, pour éviter les problèmes on ajoute un point 0 Rp, qui
        # n'aura pas d'impacte sur les barycentres muu
        while [] in S:
            S.remove([])
            S.append([np.zeros(p)])

        # on calcule le déplacement moyen des barycentres delta_muu pour la condition d'arrêt ainsi que les nouveaux
        # barycentres muu
        delta_muu = muu - barycentre(S)
        muu = barycentre(S)
        # print(norme(delta_muu))

        if show:
            [plt.scatter(muuk[0], muuk[1], c='r', s=100) for muuk in muu]
            plt.pause(0.2)

    # --------------------------------------------- fin de l'algorithme ----------------------------------------------
    # on calcule l'efficaciter du partionnement S en calculant Val un scalaire
    Val = 0
    for j in range(k):
        for i in range(len(S[j])):
            Val += norme(S[j][i] - muu[j])**2

    # on affiche tout les points si show == True avec la couleur de leurs catégories respectives
    if show:
        plt.title(f'Val = {Val}')
        plt.show()

    # si save == True on enregiste le plot avec comme nom la qualité de la partition S donné par val
    if save:
        [[plt.scatter(S[j][i][0], S[j][i][1], c=color[j]) for i in range(len(S[j]))] for j in range(k)]
        [plt.scatter(muuk[0], muuk[1], c='r', s=100) for muuk in muu]

        plt.title(f'Val = {Val}')
        plt.savefig(str(Val) + '.png')
        plt.close()

    # on retourne le partitionnement S trouver par l'algorithme des k-moyens
    return S


#%% Masque

def choixpts(img, nbpts):
    """
    Permet de choisir une liste de coordonnées sur une image

    Parameters
    ----------
    img : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels
    nbpts : TYPE int
        DESCRIPTION. nombre de points à choisir sur l'image'

    Returns
    -------
    L : TYPE list 
        DESCRIPTION. Liste des coordonnées

    """
    L = []
    while len(L) < nbpts:   #Boucle while pour être sûr de prendre des points différents/optimisation possible avec .pop en sortant les valeurs déjà utilisées
        couple = (int(np.random.randint(low=1, high=img.shape[0])-2), int(np.random.randint(low=1, high=img.shape[1])-2))   #On génère le couple d'indices
        if couple not in L:
            L.append(couple)
    return L


def Moyenne_pixel(img, coord_x, coord_y, couche):
    """
    Fonction peu élégante permettant de calculer la moyenne d'intensité d'une couleur pour un pixel

    Parameters
    ----------
    img : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels
    coord_x : TYPE int 
        DESCRIPTION. La coordonnée en x
    coord_y : TYPE int
        DESCRIPTION. La coordonnée en y
    couche : TYPE int 
        DESCRIPTION. la couche rouge, verte ou bleu 

    Returns
    -------
    TYPE float
        DESCRIPTION. Moyenne d'intensité d'un couleur pour un pixel en prenant les pixels autour de lui

    """
    p_hg = img[coord_x-1, coord_y+1, couche]    #On prends la valeur de chaque pixel
    p_hm = img[coord_x, coord_y+1, couche]
    p_hd = img[coord_x+1, coord_y+1, couche]
    p_cg = img[coord_x-1, coord_y, couche]
    p_cm = img[coord_x, coord_y, couche]
    p_cd = img[coord_x+1, coord_y, couche]
    p_bg = img[coord_x-1, coord_y-1, couche]
    p_bm = img[coord_x, coord_y-1, couche]
    p_bd = img[coord_x+1, coord_y-1, couche]
    sum_h = int(p_hg) + int(p_hm) + int(p_hd)
    sum_c = int(p_cg) + int(p_cm) + int(p_cd)
    sum_b = int(p_bg) + int(p_bm) + int(p_bd)
    return (sum_h + sum_c + sum_b)/9 #On fait la moyenne


def data_pixels(img, L):
    """
    Permet de générer une matrice contenant les données sur les intensités de pixels

    Parameters
    ----------
    img : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels
    L : TYPE list 
        DESCRIPTION. Liste des coordonnées

    Returns
    -------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions contenant les intensités et instensités moyennes des pixels choisis aléatoirement

    """
    data_img = np.zeros((len(L), 8))    #On crée notre matrice data_img
    for i in range(len(L)):
        C = np.zeros(8)
        C[0], C[1] = L[i][0], L[i][1]   #On associe les coordonnées
        C[2], C[3], C[4] = img[L[i][0], L[i][1], 0], img[L[i][0], L[i][1], 1], img[L[i][0], L[i][1], 2] #On associe les intensités des pixels
        C[5] = Moyenne_pixel(img, L[i][0], L[i][1], 0) #On associe les intensités moyenne des pixels
        C[6] = Moyenne_pixel(img, L[i][0], L[i][1], 1)
        C[7] = Moyenne_pixel(img, L[i][0], L[i][1], 2)
        data_img[i, :] = C #on ajoute la ligne
    return data_img

def ACPimg(data_img):
    """
    Permet de calculer une projection de nos données issues d'une image'

    Parameters
    ----------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions contenant les intensités et instensités moyennes des pixels choisis aléatoirement

    Returns
    -------
    Xq : TYPE np.array 
        DESCRIPTION. Matrice de la projection de nos données 
    qkaiser : TYPE int
        DESCRIPTION. nombre de directions principales obtenues avec la règle de kaiser

    """
    return ACP(data_img, 2)


def ACPimg_pond(data_img, W):
    """
    Permet de calculer une projection pondérée de nos données issues d'une image'

    Parameters
    ----------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions pondérée contenant les intensités et instensités moyennes des pixels choisis aléatoirement

    Returns
    -------
    Xq : TYPE np.array 
        DESCRIPTION. Matrice pondérée de la projection de nos données 
    qkaiser : TYPE int
        DESCRIPTION. nombre de directions principales obtenues avec la règle de kaiser

    """
    return ACPpond(data_img, 2, W)


def Kmoyimg(data_img, qkaiser):
    """
    Permet de calculer et créer des catégories à partir de données

    Parameters
    ----------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions pondérée contenant les intensités et instensités moyennes des pixels choisis aléatoirement
    qkaiser : TYPE int
        DESCRIPTION. Le nombre de directions principales à prendre en compte car elle contiennent de l'information

    Returns
    -------
    TYPE tuple
        DESCRIPTION. renvoie la matrice composée des barycentres et leurs catégories'


    """
    return Kmoy2(data_img, qkaiser, 1*10**-16)


def Masque(S, img):
    """
    Permet de créer un masque en coloriant les pixels d'une certaine couleur en fonction de leru catégories' 

    Parameters
    ----------
    S : TYPE np.array
        DESCRIPTION. Tableau par blocs contenant les catégories et les pixels 
   img : TYPE np.array
       DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    Returns
    -------
    imgmasqueponctuel : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    """
    N = int(S[-1, -1])  #On récupère le nombre de catégories
    palette = sns.color_palette(None, N)    #Création de N couleurs différentes
    imgmasqueponctuel = np.zeros(img.shape)
    for i in range(S.shape[0]):
        imgmasqueponctuel[int(S[i][0]), int(S[i][1]),0] = palette[int(S[i, -1])-1][0]*255 #On colorie chaque pixels de la couleur de sa catégorie
        imgmasqueponctuel[int(S[i][0]), int(S[i][1]),1] = palette[int(S[i, -1])-1][1]*255
        imgmasqueponctuel[int(S[i][0]), int(S[i][1]),2] = palette[int(S[i, -1])-1][2]*255
    return imgmasqueponctuel


def RemplissageMasque(imgmasqueponctuel):
    """
    algorithme d'impainting permet de "peindre" à partir d'un masque'

    Parameters
    ----------
    imgmasqueponctuel : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    Returns
    -------
    imgmasqueponctuel : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    """
    mask = cv2.cvtColor(imgmasqueponctuel, cv2.COLOR_BGR2GRAY)  #Conversion en niveau de gris
    mask[mask > 0] = 255
    mask = 255*np.ones(np.shape(mask))-mask #on applique le masque
    imgmasque = cv2.inpaint(imgmasqueponctuel, np.uint8(mask), 3, cv2.INPAINT_NS)
    return imgmasque
 