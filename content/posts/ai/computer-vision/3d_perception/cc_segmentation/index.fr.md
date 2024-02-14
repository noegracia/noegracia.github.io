---
title: "3DP-TP-00 | Segmentation de nuages de points 3D par capteur à lumière structurée RGB-D avec CloudCompare"
date: 2023-10-03T08:06:25+06:00
description: "Contrôle qualité d'objets 3D."
summary: "Contrôle qualité d'objets 3D."

math: true 
highlight: true
hightlight_languages: ["python","bash"]

authors: ["Claire Labit-Bonis"]

hero: featured.png

tags: ["teaching", "3D Perception"]

menu:
  sidebar:
    name: Segmentation
    identifier: cc-segmentation
    parent: 3d-perception
    weight: 10
---
## Téléchargement des fichiers
Les fichiers du TP peuvent être téléchargés sur votre page de cours dans Moodle, ou *via* [ce lien](files/files.zip).

## Définition des objectifs {#anchor-step-0}
Ce TP vise à effectuer une segmentation 3D et un contrôle de forme sur des objets polyédriques, c'est-à-dire à vérifier si ces objets sont corrects par rapport à un modèle géométrique de référence et/ou présentent des défauts (trous, résidus, etc.).

{{< img src="images/objets_a_comparer.png" align="center" title="Objets à comparer" >}}
{{< vs 1 >}}

Pour cela, il faut au préalable construire ce modèle de référence à partir d'une image RGB-D d'un objet sans défaut. Ensuite, pour toute vue d'un objet inconnu (dit "de test"), nous devons le segmenter à partir de l'arrière-plan et le comparer avec le modèle de référence. Ce processus de vérification de la forme doit être indépendant du point de vue et,
exige donc l'enregistrement de chaque nuage de points associé par rapport à celui de référence.

{{< alert type="info" >}}
Nous proposons de décomposer cet objectif en trois étapes :

- [étape 1](#anchor-step-1) : **extraire les points 3D** des objets à comparer (à la fois objets de référence et de test) en supprimant tous les points de la scène n'appartenant pas à l'objet. Pour éviter un processus redondant, cette étape sera à réaliser seulement sur la scène de référence contenue dans `data01.xyz` ; cela a déjà été réalisé sur les objets à contrôler, et stocké dans les fichiers `data02_object.xyz` et `data03_object.xyz`.
- [étape 2](#anchor-step-2) : **enregistrer** les points de chaque objet de test vers le modèle de référence afin de les comparer *i.e.* aligner leurs nuages de points 3D respectifs sur le repère de coordonnées de référence.
- [étape 3](#anchor-step-3) : **comparer** les modèles de contrôle et de référence et conclure sur les potentiels défauts des modèles de contrôle.
{{< /alert >}}

## ***Etape 1*** : extraction du modèle 3D de la scène de référence

La première étape du TP consiste à extraire le nuage de points du modèle de référence à partir de la scène RGB-D acquise avec une Kinect :

{{< img src="images/extraction_objet.png" align="center" title="Extraction du modèle de référence" >}}
{{< vs 1 >}}

Cette étape vise à calquer une surface plane sur le plan du sol, et à ne garder que la boite du centre en calculant la distance de chacun de ses points par rapport à ce plan et en y appliquant un seuil de filtrage.


Pour cela, ouvrez CloudCompare (le logiciel principal, pas le viewer) et importez les points de la scène `data01.xyz`. Sélectionnez le nuage en cliquant dessus dans le *workspace*.
A l'aide de l'outil de segmentation (**Edit > Segment**, ou bien directement le raccourci "ciseaux" dans la barre des raccourcis), divisez le nuage en trois sous-ensembles afin d'en extraire le plan du sol et une zone grossière autour de la boite.
Le résultat obtenu est illustré par la figure suivante :

{{< img src="images/extraction_modele.gif" align="center" title="Division de la scène en trois nuages" >}}
{{< vs 1 >}}

{{< alert type="warning" >}}
Dans CloudCompare, pour travailler sur un nuage de points, il faut que la ligne lui correspondant soit sélectionnée dans le *workspace*. Vous savez si le nuage est sélectionné lorsqu'une boite jaune s'affiche autour.

Le fait de cocher la case ne sélectionne pas le nuage, elle le rend simplement visible/invisible dans l'affichage.
{{< /alert >}}

Créez une surface calquée sur le nuage du plan du sol à l'aide de l'outil **Tools > Fit > Plane**. 
En sélectionnant le plan nouvellement créé et le nuage qui contient la boite, il est maintenant possible de calculer, pour chacun des points de ce nuage, sa distance au plan à l'aide de l'outil **Tools > Distances > Cloud/Mesh Distance** :

{{< img src="images/fit_plane_compute_distance.png" align="center" title="Surface au plan et distance" >}}
{{< vs 1 >}}

L'outil de distance ajoute un quatrième champ à chacun des points du nuage : la distance nouvellement calculée. En allant dans les propriétés du nuage, filtrez les points par rapport à ce champ scalaire pour ne garder que les points appartenant à la boite :

{{< img src="images/filter_by_value.gif" align="center" title="Filtrage par la distance au plan" >}}
{{< vs 1 >}}

En cliquant sur *split*, deux nuages sont créés, correspondant aux deux côtés du filtrage :

{{< img src="images/split.png" align="center" title="Boite extraite" >}}
{{< vs 1 >}}

Assurez-vous que le nuage nouvellement créé contient environ 10,000 points (le nombre de points est accessible dans le panneau des propriétés sur la gauche).

Sélectionnez seulement le nuage de la boite avant de l'enregistrer au format ASCII Cloud sous le nom `data01_segmented.xyz` dans le dossier `data` du TP. 

{{< alert type="info" >}}
Par précaution, sauvegardez votre projet CloudCompare : pensez à **sélectionner tous les nuages de points**, et à sauvegarder le projet au format CloudCompare.
{{< /alert >}}

## ***Etape 2*** : enregistrement des points 3D

Si vous avez ouvert les scènes complètes `data02.xyz` et `data03.xyz` dans CloudCompare, vous aurez remarqué que chaque scène a été prise d'un point de vue légèrement différent, et que les objets eux-mêmes ont bougé :

{{< img src="images/points_vue_diff.gif" align="center" title="Points de vue différents des scènes" >}}
{{< vs 1 >}}

Pour pouvoir comparer les modèles entre eux, on propose de les superposer et de calculer leur distance cumulée point à point. Plus cette distance est faible, plus les modèles se superposent et se ressemblent ; plus elle est grande, plus les modèles diffèrent.
L'exemple suivant montre la superposition du modèle correct sur le modèle de référence précédemment extrait :

{{< img src="images/plot_after_icp.png" align="center" title="Application d'ICP au modèle correct" >}}
{{< vs 1 >}}

Le fait de transformer les points d'un modèle via une matrice de rotation/translation pour venir le superposer sur un autre nuage s'appelle *l'enregistrement des points*. 
L'algorithme ***Iterative Closest Point*** permet cet enregistrement, et nous proposons de l'utiliser en Python. 
Le code à modifier se situe uniquement dans `qualitycheck.py`, l'objectif étant d'appliquer ICP à la fois sur le modèle correct `data02_object.xyz`, et sur le modèle défectueux `data03_object.xyz`.

### Chargement des modèles
La première partie du code charge les modèles `.xyz` extraits avec CloudCompare, stocke le modèle de référence dans la variable `ref` et le modèle à comparer dans la variable `data`.
Pour exécuter le code soit sur `data02_object`, soit sur `data03_object`, il suffit de commenter la ligne correspondante.

```python
# Load pre-processed model point cloud
print("Extracting MODEL object...")
model = datatools.load_XYZ_data_to_vec('data/data01_segmented.xyz')[:,:3]

# Load raw data point cloud
print("Extracting DATA02 object...")
data02_object = datatools.load_XYZ_data_to_vec('data/data02_object.xyz')

# Load raw data point cloud
print("Extracting DATA03 object...")
data03_object = datatools.load_XYZ_data_to_vec('data/data03_object.xyz')

ref = model
data = data02_object
# data = data03_object
```

### Appel à ICP

La deuxième partie du code consiste à coder l'appel à la fonction `icp` de la librairie `icp`... 

```python
##########################################################################
# Call ICP:
#   Here you have to call the icp function in icp library, get its return
#   variables and apply the transformation to the model in order to overlay
#   it onto the reference model.

matrix = np.eye(4,4)        # Transformation matrix returned by icp function
errors = np.zeros((1,100))  # Error value for each iteration of ICP
iterations = 100            # The total number of iterations applied by ICP
total_time=0                # Total time of convergence of ICP

# ------- YOUR TURN HERE -------- 

# Draw results
fig = plt.figure(1, figsize=(20, 5))
ax = fig.add_subplot(131, projection='3d')
# Draw reference
datatools.draw_data(ref, title='Reference', ax=ax)

ax = fig.add_subplot(132, projection='3d')
# Draw original data and reference
datatools.draw_data_and_ref(data, ref=ref, title='Raw data', ax=ax)
```

...et à stocker le retour de la fonction dans les variables `T`, `errors`, `iterations` et `total_time` comme défini par l'en-tête de définition de la fonction dans le fichier `icp.py` :

```python
def icp(data, ref, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        errors: Euclidean distances (errors) for max_iterations iterations in a (max_iterations+1) vector. distances[0] is the initial distance.
        i: number of iterations to converge
        total_time : execution time
    '''
```

### Transformation du modèle 

La matrice `T` de transformation issue d'ICP est la matrice de passage homogène permettant de calquer le modèle `data`, passé en paramètre de la fonction `icp`, sur le modèle `ref`.
Pour rappel, l'application d'une matrice homogène pour transformer un ensemble de points d'un repère initial \\(\mathcal{R_i}\\) vers un repère final \\(\mathcal{R_f}\\) s'effectue de la manière suivante :

$$P_f^{(4 \times N)} = T^{(4 \times 4)} . P_i^{(4 \times N)}$$

Dans le code, la troisième partie consiste donc à appliquer la matrice de transformation au modèle à comparer. Un exemple de l'application d'une matrice de rotation homogène à une autre matrice est donné ci-après :

```python
# EXAMPLE of how to apply a homogeneous transformation to a set of points
''' 
# (1) Make a homogeneous representation of the model to transform
homogeneous_model = np.ones((original_model.shape[0], 4))   ##### Construct a [N,4] matrix
homogeneous_model[:,0:3] = np.copy(original_model)          ##### Replace the X,Y,Z columns with the model points
# (2) Construct the R|t homogeneous transformation matrix / here a rotation of 36 degrees around x axis
theta = np.radians(36)
c, s = np.cos(theta), np.sin(theta)
homogeneous_matrix = np.array([[1, 0, 0, 0],
                               [0, c, s, 0],
                               [0, -s, c, 0],
                               [0, 0, 0, 1]])
# (3) Apply the transformation
transformed_model = np.dot(homogeneous_matrix, homogeneous_model.T).T
# (4) Remove the homogeneous coordinate
transformed_model = np.delete(transformed_model, 3, 1)
'''
```

{{< alert type="info" >}}
La variable `original` est un tableau de taille \\(N \times 3\\), \\(N\\) étant le nombre de points du modèle et 3 ses coordonnées \\(X\\), \\(Y\\) et \\(Z\\).

Il faut veiller à lui ajouter une coordonnée homogène et appliquer les transposées nécessaires pour que la multiplication de matrices fonctionne.
Aidez-vous de l'exemple donné dans le code pour réaliser cette étape.
{{< /alert >}}

Vous pouvez ensuite afficher le résultat en décommentant et complétant la ligne `datatools.draw_data...`.

### Affichage de l'erreur
Décommentez et affichez l'erreur dans la dernière partie du code, en changeant les "..." par les variables correspondantes :

```python
# Display error progress over time
# **************** To be uncommented and completed ****************
# fig1 = plt.figure(2, figsize=(20, 3))
# it = np.arange(0, len(errors), 1)
# plt.plot(it, ...)
# plt.ylabel('Residual distance')
# plt.xlabel('Iterations')
# plt.title('Total elapsed time :' + str(...) + ' s.')
# plt.show()
```

## ***Etape 3*** : comparaison des modèles

Comparez l'application d'ICP sur les modèles `data02` et `data03`, remarquez l'évolution de l'erreur et les différences de valeurs.
Que représente cette erreur ? Que peut-on dire des deux modèles ? En vous appuyant sur les erreurs, quel seuil de décision pourriez-vous choisir pour déterminer si un modèle est défectueux ou non ?

### ICP dans CloudCompare

L'algorithme ICP peut également être utilisé directement dans CloudCompare. Ouvrez-le et importez `data01_segmented.xyz`, `data02_object.xyz` et `data03_object.xyz`.

Sélectionnez par exemple les nuages de `data01_segmented` et `data02_object`, utilisez l'outil **Tools > Registration > Fine registration (ICP)**. Assurez-vous que la référence est bien `data01` et appliquez ICP.
Son exécution vous renvoie la matrice de transformation calculée par l'algorithme, et l'applique à l'objet.

{{< img src="images/registration_cc.png" align="center" title="ICP dans CloudCompare" >}}
{{< vs 1 >}}

On peut ainsi, toujours en sélectionnant les deux nuages, calculer la distance entre les points avec **Tools > Distance > Cloud/Cloud Distance**. Assurez-vous que la référence est bien `data01` et cliquez sur OK/Compute/OK.
Sélectionnez `data02_object` et affichez l'histogramme de ses distances au nuage de référence *via* **Edit > Scalar fields > Show histogram**.

{{< img src="images/histogram_dists.png" align="center" title="Histogramme des distances point à point de data02 vs. data01" >}}
{{< vs 1 >}}

Faites la même chose avec `data03_object` et comparez les histogrammes. Comment les interprétez-vous ? Comment pouvez-vous les comparer ?

TP crée par notre proffesseur [Claire Labit-Bonis](https://clairelabitbonis.github.io/).