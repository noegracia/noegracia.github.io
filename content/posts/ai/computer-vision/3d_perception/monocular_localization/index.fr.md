---
title: "3DP-TP-01 | Localisation monoculaire par PnL itérative"
date: 2022-07-16T08:06:25+06:00
description: "Iterative estimation of a camera extrinsic parameters."
summary: "Iterative estimation of a camera extrinsic parameters."

math: true 
highlight: true
hightlight_languages: ["python","bash"]

authors: ["Claire Labit-Bonis"]

hero: featured.png

tags: ["Teaching", "3D Perception"]

menu:
  sidebar:
    name: Localisation mono.
    identifier: monocular-localization
    parent: 3d-perception
    weight: 20
---
<!-- #TODO: expliquer dmatPos*matPos, lambda, détailler le calcul de matrice intrinsic inverse et stockage des segments 
#l1-l2. Pourquoi ne pas utiliser Ni.(P2i-P1i) plutôt que Ni.P1i + Ni.P2i ?
#Developpement de Taylor ordre 1 pour F(X) courant : tangente à la fonction en ce nouveau point.
# Transpoint, changer les noms des variables : P_i => P, P_f => P_transformed ?
# Modifier les matrices modA, App3d, [0,1,2], [0,3,5] etc.
# Pourquoi err/2N ? Moyenne? -->

>Cet exercice consiste à remplir des fonctions ou des morceaux vides dans le fichier `localisation.py` : 
>- [étape 1](#anchor-step-1) : `perspective_projection` et `transform_and_draw_model`, 
>- [étape 2](#anchor-step-2) : `calculate_normal_vector` et `calculate_error`
>- [étape 3](#anchor-step-3) : `partial_derivatives`,
>- [étape 4](#anchor-step-4) : transformation vers un deuxième point de vue.
>
>Chaque fonction et le rôle qu'elle remplit sont décrits dans cet article.

{{< alert type="danger" >}}
Ne vous jetez pas dans le codage des fonctions dès la partie "définition des objectifs", qui n'est qu'une explication théorique globale pour présenter le sujet. Chaque étape et les fonctions qui qui sont propres sont décrites en détail dans les parties correspondantes : [étape 1](#anchor-step-1), [étape 2](#anchor-step-2), [étape 3](#anchor-step-3), [étape 4](#anchor-step-4).
{{< /alert >}}

## Téléchargement des fichiers
Les fichiers du TP peuvent être téléchargés sur votre page de cours dans Moodle, ou *via* [ce lien](files/files.zip). 

## Définition des objectifs {#anchor-step-0}
L'objectif de cet exercice est de trouver la transformation optimale à appliquer à un modèle 3D exprimé en centimètres dans un repère *objet* \\(\mathcal{R\_o} (X,Y,Z)\\) (parfois appelé repère *monde* \\(\mathcal{R\_w}\\)) afin de le dessiner dans une image 2D exprimée en pixels dans un repère *image* \\(\mathcal{R\_i} (u,v)\\). Le modèle 3D a été téléchargé sur [free3d](https://free3d.com) et modifié dans [Blender](https://www.blender.org/) pour les besoins de l'exercice. Ses points et arêtes ont respectivement été exportés *via* un script Python dans les fichiers `pikachu.xyz` et `pikachu.edges`.

{{< img src="images/blender_pikachu.png" align="center" title="Modèle 3D réalisé dans Blender" >}}
{{< vs 1 >}}

`pikachu.xyz` contains 134 lines and 3 columns, corresponding to the 134 model points and their three \\(x, y, z\\) coordinates.

`pikachu.edges` contient 246 lignes et 2 colonnes, correspondant aux 246 arêtes du modèle et aux 2 indices de leurs extrémités (la première arête est définie par son premier point à l'indice 2 et par son deuxième point à l'indice 0 dans `pikachu.xyz`).

{{< img src="images/edges_xyz.png" align="center" title="Fichiers de coordonnées et arêtes" >}}
{{< vs 1 >}}

Par "transformation", on entend une rotation et une translation de la scène
sur les axes \\(x\\), \\(y\\) et \\(z\\) des points dans \\(\mathcal{R\_o}\\).

Dans notre cas, on veut augmenter la réalité capturée par la caméra et positionner le modèle de Pikachu sur le cube bleu ciel de l'image ci-dessous, en le faisant parfaitement *matcher* avec le cube virtuel sur lequel il repose.

{{< img src="images/final_objective.png" align="center" title="Objectif à atteindre" >}}
{{< vs 1 >}}

<details>
<summary><strong>Paramètres intrinsèques. <span style="color:pink">Cliquez pour étendre</span></strong></summary> 

On utilise le modèle de caméra sténopé qui permet de réaliser cette opération en deux transformations successives :\\(\mathcal{R\_o} \rightarrow \mathcal{R\_c}\\) puis \\(\mathcal{R\_c} \rightarrow \mathcal{R\_i}\\). En plus des repères \\(\mathcal{R\_o}\\) et \\(\mathcal{R\_i}\\), il faut donc aussi considérer le repère caméra \\(\mathcal{R\_c}\\).

>La ressource *[Modélisation et calibrage d'une caméra](http://www.optique-ingenieur.org/fr/cours/OPI_fr_M04_C01/co/Grain_OPI_fr_M04_C01_2.html)* décrit en détail le modèle sténopé et peut aider à la compréhension de l'exercice.

{{< img src="images/goal.png" align="center" title="Changements de repères" >}}
{{< vs 1 >}}

> Wikipédia définit le [modèle sténopé](https://en.wikipedia.org/wiki/Pinhole_camera_model) comme décrivant *la relation mathématique entre les coordonnées d'un point dans un espace en trois dimensions et sa projection dans le plan image d'une caméra sténopé i.e. une caméra dont l'ouverture est décrite comme un point et qui n'utilise pas de lentille pour focaliser la lumière*. 

Le passage entre les repères \\(\mathcal{R\_c}\\) anetd \\(\mathcal{R\_i}\\) se fait à l'aide des paramètres **intrinsèques** de la caméra. Ces coefficients \\((\alpha\_u, \alpha\_v, u\_0, v\_0)\\) sont ensuite stockés dans une matrice de passage homogène \\(K\_{i \leftarrow c}\\) de sorte que l'on peut décrire la relation \\(p\_i = K\_{i \leftarrow c}.P\_c\\) comme étant :

$$
\begin{bmatrix}
u\\\\v\\\\1
\end{bmatrix}\_{\mathcal{R}\_i} = s. 
\begin{bmatrix}
\alpha\_u & 0 & u\_0\\\\
0 & \alpha\_v & v\_0\\\\
0 & 0 & 1
\end{bmatrix} . 
\begin{bmatrix}
X\\\\
Y\\\\
Z
\end{bmatrix}\_{\mathcal{R}\_c}
$$

avec :

- \\(p\_i\\) (à gauche) le point exprimé en pixels dans le repère 2D image \\(\mathcal{R\_i}\\),
- \\(P\_c\\) (à droite) le point exprimé en centimètres dans le repère 3D caméra \\(\mathcal{R\_c}\\),
- \\(s = \frac{1}{Z}\\),
- \\(\alpha_u = k_x f\\), \\(\alpha_v = k_y f\\) :
    - \\(k_x = k_y\\) le nombre de pixels par millimètre du capteur dans les directions \\(x\\) et \\(y\\) -- l'égalité n'étant vraie que si les pixels sont carrés,
    - \\(f\\) la distance focale.
- \\(u_0\\) et \\(v_0\\) les centres de l'image en pixels dans \\(\mathcal{R}_i\\).

Dans notre cas, l'image a été capturée par un Canon EOS 700D dont la taille du capteur est de \\(22.3\times14.9 mm\\). La taille de l'image étant de \\(720\times480 px\\) et la distance focale de \\(18mm\\), on en déduit les paramètres \\(\alpha\_u = 581.1659\\), \\(\alpha\_v = 579.8657\\), \\(u\_0 = 360\\) et \\(v\_0 = 240\\) qui sont stockés dans le fichier `calibration_parameters.txt`.
</details>

<details>
<summary><strong>Paramètres extrinsèques. <span style="color:pink">Cliquez pour étendre</span></strong></summary> 

Pour atteindre notre objectif de départ, c'est-à-dire afficher dans l'image en pixels notre modèle 3D virtuel, nous devons estimer les coefficients de la transformation \\(\mathcal{R\_o}\rightarrow\mathcal{R\_c}\\) permettant d'établir la relation \\(P\_c=M\_{c\leftarrow o}.P\_o\\), avec :

- \\(M\_{c\leftarrow o}=\big[ R\_{\alpha\beta\gamma} | T \big]\\) la matrice de transformation homogène composée des angles d'Euler et de la translation selon les axes \\(x\\), \\(y\\) et \\(z\\) du repère.

- \\(R\_{\alpha\beta\gamma}\\) la matrice de rotation résultat de l'application successive (et donc de la multiplication entre elles) des matrices de rotation \\(R\_\gamma\\), \\(R\_\beta\\) et \\(R\_\alpha\\) :

$$R\_\alpha = \begin{bmatrix}1 & 0 & 0\\\\0 & \cos \alpha & -\sin \alpha\\\\0 & \sin \alpha & \cos \alpha\end{bmatrix}$$

$$R\_\beta = \begin{bmatrix}\cos \beta & 0 & -\sin \beta\\\\0 & 1 & 0\\\\\sin \beta & 0 & \cos \beta\end{bmatrix}$$

$$R\_\gamma = \begin{bmatrix}\cos \gamma & -\sin \gamma & 0\\\\\sin \gamma & \cos \gamma & 0\\\\0 & 0 & 1\end{bmatrix}$$

Avec les bons paramètres extrinsèques \\((\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\), le modèle 3D dans son repère \\(\mathcal{R\_o}\\) pourra être transformé jusqu'à être exprimé dans \\(\mathcal{R\_c}\\) puis dans \\(\mathcal{R\_i}\\), en respectant la relation suivante pour chaque point \\(P_o\\) :

$$ p\_i = K\_{i \leftarrow c} M\_{c\leftarrow o} P\_o$$

$$\begin{bmatrix}
u\\\\
v\\\\
1
\end{bmatrix}\_{\mathcal{R}\_i} = s. 
\begin{bmatrix}
\alpha\_u & 0 & u\_0\\\\
0 & \alpha\_v & v\_0\\\\
0 & 0 & 1
\end{bmatrix} \bigodot 
\begin{bmatrix}
r\_{11} & r\_{12} & r\_{13} & t\_x\\\\
r\_{21} & r\_{22} & r\_{23} & t\_y\\\\
r\_{31} & r\_{32} & r\_{33} & t\_z\\\\
0 & 0 & 0 & 1
\end{bmatrix} . 
\begin{bmatrix}
X\\\\
Y\\\\
Z\\\\
1
\end{bmatrix}\_{\mathcal{R}\_o}
$$

{{< alert type="warning" >}}
L'opérateur \\(\bigodot\\) indique la multiplication des termes après avoir enlevé la coordonnée homogène de \\(M\_{c\leftarrow o} P\_o\\). Il est simplement là pour que les tailles de matrice soient correctes.
{{< /alert >}}
</details>
</br>

>Les paramètres intrinsèques sont connus ; pour trouver les paramètres extrinsèques nous utiliserons une méthode de localisation (*i.e.* d'estimation des paramètres) dite *par recalages successifs*, appelée **Perspective-n-Lignes**.

### Estimation des paramètres extrinsèques

On démarre d'une estimation initiale grossière des paramètres \\((\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\). Cette estimation initiale est plus ou moins juste en fonction de la connaissance que l'on a *a priori* de la scène, de l'objet, de la position de la caméra.

Dans notre cas, les paramètres initiaux ont pour valeurs \\(alpha = -2.1\\), \\(beta = 0.7\\), \\(gamma = 2.7\\), \\(t\_x = 3.1\\), \\(t\_y = 1.3\\) et \\(t\_z = 18\\), c'est-à-dire que l'on sait avant même de commencer que le modèle 3D devra subir une rotation d'angles \\((-2.1, 0.7, 2.7)\\) autour de ses trois axes, et qu'il devra se décaler d'environ \\(3cm\\) en \\(x\\), \\(1.5cm\\) en \\(y\\) and \\(18cm\\) en \\(z\\).

>Cette connaissance *a priori* de la scène vient du fait que l'on est capable, en tant qu'humain, d'évaluer la distance de l'objet réel par rapport au capteur seulement en regardant l'image. 

### Utilisation du programme

Au lancement du script `localisation.py`, deux figures s'ouvrent : l'une représente une scène réelle capturée par la caméra, l'autre représente le modèle virtuel à transformer et dont l'origine du repère est matérialisée par un point rouge. L'objectif étant de venir calquer la boite sur laquelle repose le Pikachu par-dessus le cube bleu ciel posé sur la table, il faut dans un premier temps sélectionner cinq arêtes appartenant à la boite réelle (par *click* souris sur les extrémités des arêtes), puis sélectionner dans le même ordre les arêtes correspondantes sur le modèle 3D (par *click* souris au milieu des arêtes). Le nombre de segments à sélectionner (par défaut 5), est fixé dès le départ dans la variable `nb_segments`.

{{< img src="images/edge_selection.png" align="center" title="Sélection des arêtes dans l'image et dans le modèle 3D" >}}
{{< vs 1 >}}

De cette manière, on pourra calculer l'erreur commise sur l'estimation des paramètres extrinsèques en comparant la distance entre les arêtes sélectionnées dans l'image et celles du modèle transformé. Ce critère d'erreur est décrit à l'[étape 2](#anchor-step-2).

## ***Etape 1*** : afficher le modèle dans \\(\mathcal{R_i}\\) {#anchor-step-1}

Dans le programme principal, les points du modèle sont stockés dans `model3d_Ro`, une matrice de taille \\([246\times6]\\) correspondant aux 246 arêtes du modèle, chacune définie par les 6 coordonnées de ses deux points \\(P_1(x_1, y_1, z_1)\\) et \\(P_2(x_2, y_2, z_2)\\). La matrice de transformation \\(M\_{c\leftarrow o}\\) est stockée dans `extrinsic_matrix` et les paramètres intrinsèques de la caméra sont stockés dans `intrinsic_matrix`.

Pour visualiser le modèle 3D dans l'image 2D et se faire une idée de la justesse de notre estimation, il faut coder la fonction `tranform_and_draw_model` qui permet d'appliquer la transformation \\(K\_{i \leftarrow c} M\_{c\leftarrow o}\\) à chaque point \\(P\_o\\) d'un ensemble d'arêtes' `edges_Ro`, avec une matrice `intrinsic`, et une matrice `extrinsic` pour finalement afficher le résultat dans une figure `fig_axis` :

``` python
def transform_and_draw_model(edges_Ro, intrinsic, extrinsic, fig_axis):
    # ********************************************************************* #
    # A COMPLETER.                                                          #
    # UTILISER LES FONCTIONS :                                              #
    #   - perspective_projection                                            #  
    #   - transform_point_with_matrix                                       #
    # Input:                                                                #
    #   edges_Ro : ndarray[Nx6]                                             #
    #             N = nombre d'aretes dans le modele                        #
    #             6 = (X1, Y1, Z1, X2, Y2, Z2) les coordonnees des points   #
    #                 P1 et P2 de chaque arete                              #
    #   intrinsic : ndarray[3x3] - parametres intrinseques de la camera     #
    #   extrinsic : ndarray[4x4] - parametres extrinseques de la camera     #
    #   fig_axis : figure utilisee pour l'affichage                         #
    # Output:                                                               #
    #   Pas de retour de fonction, mais calcul et affichage des points      #
    #   transformes (u1, v1) et (u2, v2)                                    #
    # ********************************************************************* #

    # Part to replace #
    u_1 = np.zeros((edges_Ro.shape[0],1))
    u_2 = np.zeros((edges_Ro.shape[0],1))
    v_1 = np.zeros((edges_Ro.shape[0],1))
    v_2 = np.zeros((edges_Ro.shape[0],1))
    ############### 
    
    for p in range(edges_Ro.shape[0]):
        fig_axis.plot([u_1[p], u_2[p]], [v_1[p], v_2[p]], 'k')
```

L'objectif est de stocker dans les points `[u_1, v_1]` et `[u_2, v_2]` les coordonnées \\((u, v)\\) des points \\(P_1\\) et \\(P_2\\) des arêtes après transformation.

On peut découper cette fonction en deux sous-étapes : 

* la transformation \\(\mathcal{R\_o} \rightarrow \mathcal{R\_c}\\) des points \\(P\_o\\) à l'aide de la fonction `transform_point_with_matrix` fournie dans la librairie `matTools` :

    ```python
    P_c = matTools.transform_point_with_matrix(extrinsic, P_o)
    ```

* la projection \\(\mathcal{R\_c} \rightarrow \mathcal{R\_i}\\) des points nouvellement obtenus \\(P\_c\\) à l'aide de la fonction `perspective_projection` qu'il faut compléter :

    ```python
    def perspective_projection(intrinsic, P_c):
        # ***************************************************** #
        # A COMPLETER.                                          #
        # Fonction utile disponible :                           #
        #   np.dot                                              #
        # Input:                                                #
        #   intrinsic : ndarray[3x3] - parametres intrinseques  #
        #   P_c : ndarray[Nx3],                                 #
        #         N = nombre de points à transformer            #
        #         3 = (X, Y, Z) les coordonnees des points      #
        # Output:                                               #
        #   u, v : deux ndarray[N] contenant les                #
        #          coordonnees Ri des points P_c transformes    #
        # ***************************************************** #
        
        u, v = 0, 0 # Part to replace
        
        return u, v
    ```

Une fois `perspective_projection` et `transform_and_draw_model` complétées, le lancement du programme global projette le modèle dans l'image, avec les paramètres extrinsèques définis au départ.

{{< img src="images/first_rough_estimate.png" align="center" title="Première projection du modèle dans l'image" >}}
{{< vs 1 >}}

>La transformation appliquée à ces points n'est visiblement pas très bonne, le modèle ne *matche* pas la boite comme on le souhaiterait. Pour pouvoir différencier une "mauvaise" estimation des paramètres extrinsèques d'une "bonne", et ainsi pouvoir proposer de manière automatique une nouvelle estimation plus proche de notre objectif, il faut déterminer un critère d'erreur caractérisant la distance à laquelle on se trouve de cet objectif optimal.

## ***Etape 2*** : déterminer un critère d'erreur {#anchor-step-2}

Le critère d'erreur sert à évaluer la justesse de notre estimation. Plus l'erreur est grande, moins nos paramètres extrinsèques sont bons. Après avoir déterminé ce critère d'erreur, on pourra l'intégrer dans une boucle d'optimisation visant à le minimiser, et donc à avoir des paramètres extrinsèques les meilleurs possibles pour notre objectif d'appariement 2D/3D.

A titre d'exemple, la figure ci-dessous illustre cette boucle d'optimisation pour la projection de la boite virtuelle sur le cube réel. A l'itération \\(0\\), les paramètres sont grossiers, l'erreur est grande. En modifiant les paramètres on fera diminuer l'erreur jusqu'à atteindre, dans l'idéal, une erreur nulle et une transformation optimale pour une projection parfaite du modèle 3D dans l'image 2D.

{{< img src="images/extrinsic_optim.gif" align="center" title="Optimisation des paramètres extrinsèques" >}}
{{< vs 1 >}}

Dans le cadre de l'appariement 2D/3D de segments, le critère d'erreur à minimiser correspond au produit scalaire de la normale au plan d'interprétation relatif à l'appariement. En d'autres termes, l'objectif est de transformer les points du modèle de sorte que les segments sélectionnés dans l'image et ceux sélectionnés dans le modèle appartiennent au même plan exprimé dans le repère caméra \\(\mathcal{R\_c}\\).

{{< img src="images/interpretation_plan.png" align="center" title="Plan d'interprétation relatif à l'appariement de droites" >}}
{{< vs 1 >}}

Comme le montre la figure ci-dessus, on peut définir le *plan d'interprétation* comme étant formé par les segments \\(\mathcal{l\_{i \rightarrow c}^{j,1}}\\) et \\(\mathcal{l\_{i \rightarrow c}^{j,2}}\\). Dans cette notation, \\(j\\) correspond au nombre d'arêtes sélectionnées au lancement du programme (5 par défaut). Pour chaque arête sélectionnée \\(j\\) et exprimée dans le repère *image* \\(\mathcal{R_i}\\), on a donc deux segments \\(l^1\\) et \\(l^2\\). Le segment \\(l^1\\) est composé des deux extrémités \\(P\_{i \rightarrow c}^0\\) et \\(P\_{i \rightarrow c}^1\\) ; le segment \\(l^2\\) est composé des deux extrémités \\(P\_{i \rightarrow c}^1\\) et \\(P\_{i \rightarrow c}^2\\). Chacun des \\(P\_{i \rightarrow c}\\) est un point du repère *image* \\(\mathcal{R\_i}\\) transformé dans le repère *caméra* \\(\mathcal{R\_c}\\). Ils sont connus : ce sont ceux qui ont été sélectionnés par *click* souris.

Une fois ces segments calculés, on peut calculer la normale au plan d'interprétation \\(N\_c^j\\). Pour rappel :

$$N = \frac{l^1 \wedge l^2}{||l^1 \wedge l^2||}$$ 

Dans la fonction de sélection des segments `utils.select_segments()`, au fil des sélections d'arêtes, les normales sont calculées et stockées dans la matrice `normal_vectors` grâce à la fonction `calculate_normal_vector` qu'il faut compléter dans le fichier `localisation.py` :

```python
def calculate_normal_vector(p1_Ri, p2_Ri, intrinsic):
    # ********************************************************* #
    # A COMPLETER.                                              #
    # Fonctions utiles disponibles :                            #
    #   np.dot, np.cross, np.linalg.norm, np.linalg.inv         #
    # Input:                                                    #
    #   p1_Ri : list[3]                                         #
    #           3 = (u, v, 1) du premier point selectionne      #
    #   p2_Ri : list[3]                                         #
    #           3 = (u, v, 1) du deuxieme point selectionne     #
    #   intrinsic : ndarray[3x3] des intrinseques               #
    # Output:                                                   #
    #   normal_vector : ndarray[3] contenant la normale aux     #
    #                   segments L1_c et L2_c deduits des       #
    #                   points image selectionnes               #
    # ********************************************************* #

    normal_vector = np.zeros((len(p1_Ri),)) # Part to replace

    return normal_vector
```

Ensuite, pour chaque segment \\(j \in \[1, ..., 5\]\\), la distance entre le plan lié aux segments image et les points du modèle 3D peut être calculée grâce au produit scalaire des \\(N\_c^j\\) et des \\(P\_{o\rightarrow c}^j\\). Si le produit scalaire est nul, les deux vecteurs sont orthogonaux et le point \\(P\_{o\rightarrow c}^j\\) appartient bien au même plan que \\(P\_{i \rightarrow c}^j\\). Plus le produit scalaire est grand, plus les paramètres extrinsèques s'éloignent de la bonne transformation.

Pour résumer, le critère d'optimisation des paramètres est \\(\sum\_{j=1}^{2n} F^j(X)^2\\) (le carré enlève les valeurs négatives), avec \\(F^j(X) = N^{j \; \text{mod} \; 2}.P\_{o\rightarrow c}^{j \; \text{mod} \; 2, 1|2}\\), selon que l'on se trouve sur le point \\(P^{1}\\) ou \\(P^{2}\\). 

> Dans la somme qui parcourt les points de \\(j\\) à \\(2n\\), \\(j \; \text{mod} \; 2\\) est l'indice du segment pour le point courant. Autrement dit, on cumule les \\((N^i.P^{i, 1})^2\\) et \\((N^i.P^{i, 2})^2\\) pour tous les segments \\(i\\).

{{< alert type="warning" >}}
Ici, \\(X\\) désigne l'ensemble des paramètres \\((\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\) et non une coordonnée.
{{< /alert >}}

On rappelle que chaque \\(P\_{o\rightarrow c} = \big[ R\_{\alpha\beta\gamma} | T \big]. P\_o\\) ; la valeur de l'erreur dépend donc bien de la valeur des paramètres extrinsèques. A chaque passage dans la boucle d'optimisation, changer les poids de ces paramètres aura une influence sur le critère \\(F(X)\\).

La fonction `calculate_error` calcule le critère d'erreur. Elle prend en entrée le nombre `nb_segments` d'arêtes sélectionnées, les `normal_vectors`, et les `segments_Rc` sélectionnés puis transformés par la matrice des paramètres extrinsèques et exprimés dans \\(\mathcal{R_c}\\).

```python
def calculate_error(nb_segments, normal_vectors, segments_Rc):
    # ***************************************************************** #
    # A COMPLETER.                                                      #
    # Input:                                                            #
    #   nb_segments : par defaut 5 = nombre de segments selectionnes    #
    #   normal_vectors : ndarray[Nx3] - normales aux plans              #
    #                    d'interpretation des segments selectionnes     #
    #                    N = nombre de segments                         #
    #                    3 = coordonnees (X,Y,Z) des normales dans Rc   #
    #   segments_Rc : ndarray[Nx6] = segments selectionnes dans Ro      #
    #                 et transformes dans Rc                            #
    #                 N = nombre de segments                            #
    #                 6 = (X1, Y1, Z1, X2, Y2, Z2) des points P1 et     #
    #                 P2 des aretes transformees dans Rc                #
    # Output:                                                           #
    #   err : float64 - erreur cumulee des distances observe/attendu    #
    # 
    # ***************************************************************** #
    err = 0
    for p in range(nb_segments):
        # Part to replace with the error calculation
        err = err + 0
    
    err = np.sqrt(err / 2 * nb_segments)
    return err
```

> D'un point de vue mathématique, la somme est écrite pour \\(j\\) allant de \\(1\\) à \\(2n\\), *i.e.* le nombre de points. D'un point de vue implémentation et étant donnée la construction de nos variables, il est plus simple d'exprimer le critère au travers d'une somme parcourant les arêtes : \\(\sum\_{j=1}^{n} F^j(X)\\) with \\(F^{j}(X) = F^{j, 1}(X)^2 + F^{j, 2}(X)^2\\).
> 
> Ainsi, \\(F^{j, 1}(X) = N^j.P\_{o\rightarrow c}^{j, 1}\\) et \\(F^{j, 2}(X) = N^j.P\_{o\rightarrow c}^{j, 2}\\). 

## ***Etape 3*** : Estimation des paramètres par la méthode des moindres carrés ordinaires {#anchor-step-3}

<!-- [//]: # (A chaque étape, on cherche un incrément à appliquer à chacun des paramètres $X (\alpha, \beta, \gamma, t\_x, t\_y, t\_z)$. Pour l'exemple, on peut s'intéresser seulement au paramètre $\gamma$ de rotation autour de l'axe $Z$. La procédure peut alors être illustrée par la figure suivante : ![Incréments successifs des paramètres extrinsèques](images/increment_gamma.png) Le point de départ est exprimé dans son repère objet $\mathcal{R\_O}$. A l'itération $k=0$, la transformation effectuée est : $$\begin{bmatrix}X \\\\\\ Y \\\\\\ Z\end{bmatrix}\_{\mathcal{R\_C\_0}}} = \[R\_\gamma . R\_\beta . R\_\alpha | T\_{xyz}\]\_{k=0}\begin{bmatrix}X \\\\\\ Y \\\\\\ Z\end{bmatrix}\_{\mathcal{R\_{O}}}$$ Si l'on appelle $M\_{k=0}$ la matrice des paramètres extrinsèques $\[R\_\gamma . R\_\beta . R\_\alpha | T\_{xyz}\]\_{k=0}$, on aura à l'itération $k=1$ : $$ \begin{bmatrix}X \\\\\\ Y \\\\\\ Z\end{bmatrix}\_{\mathcal{R\_{C\_1}}} = \[R\_{\Delta\gamma} . R\_{\Delta\beta} . R\_{\Delta\alpha} | T\_{\Delta x, \Delta y, \Delta z}\]\_{k=1}. M\_{k=0}.\begin{bmatrix}X \\\\\\ Y \\\\\\ Z \end{bmatrix}\_{\mathcal{R\_{O}}}$$) -->

A chaque itération \\(k\\), on cherche un jeu de paramètres \\(X\_{k}(\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\) tel que le critère \\(F(X)\\) soit égal au critère pour le jeu de paramètres précédent \\(X_0\\) (avant mise à jour) incrémenté d'un delta pondéré par la jacobienne de la fonction.

On cherche donc à résoudre un système de la forme :

$$F(X) \approx F(X\_0) + J \Delta X$$

>La jacobienne \\(J\\) contient sur ses lignes les dérivées partielles de la fonction \\(F\\) pour chaque point sélectionné et selon chacun des paramètres de \\(X\\). Elle traduit la tendance du critère (montant/descendant) et la vitesse à laquelle il augmente ou diminue en fonction de la valeur de chacun de ses paramètres.

Notre objectif étant d'atteindre un critère \\(F(X) = 0\\), on traduit le problème à résoudre par :

$$
\begin{align}
0 &= F(X\_0) + J \Delta X \\\\\\
\Leftrightarrow \qquad -F(X\_0) &= J\Delta X
\end{align}
$$

L'intérêt de travailler par incréments successifs est d'avoir des valeurs d'incrément si petites par rapport à la dernière itération qu'on peut approximer la variation de ces paramètres comme étant égale à 0. Pour rappel, le critère d'erreur s'exprime de la manière suivante pour chaque segment \\(j\\) :

$$
\begin{align}
F^{j, 1}(X) &= N^j.P\_{o\rightarrow c}^{j, 1} \text{ avec } P\_{o\rightarrow c}^{j, 1} = \big[ R\_{\alpha\beta\gamma
} | T \big] . P\_o^{j, 1}\\\\\\
F^{j, 2}(X) &= N^{j}.P\_{o\rightarrow c}^{j, 2} \text{ avec } P\_{o\rightarrow c}^{j, 2} = \big[ R\_{\alpha\beta\gamma
} | T \big] . P\_o^{j, 2}
\end{align}$$

Du fait de toujours avoir des\\(\Delta X \approx 0\\), le calcul de la matrice jacobienne se retrouve considérablement simplifié, puisque les dérivées partielles \\((\frac{\partial F^j}{\partial \alpha}, \frac{\partial F^j}{\partial \beta}, \frac{\partial F^j}{\partial \gamma}, \frac{\partial F^j}{\partial t\_x}, \frac{\partial F^j}{\partial t\_y}, \frac{\partial F^j}{\partial t\_z})\\) sont les mêmes à chaque itération.

Le détail de la dérivation est donné pour le premier paramètre \\(\alpha\\), les suivants sont à démontrer :

$$\begin{align}
    \frac{\partial F^j}{\partial \alpha} &= N^j . [R\_\gamma . R\_\beta . \frac{\partial R\_\alpha}{\partial \alpha} | T] . P\_o^j\\\\\\
\end{align}$$

{{< alert type="info" >}}
\\(R\_\gamma\\) et \\(R\_\beta\\) disparaissent de la dérivation car pour \\(\gamma \approx 0\\) et \\(\beta \approx 0\\), on a :

$$
R_{\gamma \approx 0} = \begin{bmatrix} 
\cos 0 & -\sin 0 & 0\\\\\\ 
\sin 0 & \cos 0 & 0\\\\\\ 
0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 
1 & 0 & 0\\\\\\ 
0 & 1 & 0\\\\\\ 
0 & 0 & 1 \end{bmatrix} = I_3
$$

$$
R_{\beta \approx 0} = \begin{bmatrix} 
\cos 0 & 0 & -\sin 0\\\\\\
0 & 1 & 0\\\\\\
\sin 0 & 0 & \cos 0 \end{bmatrix} = \begin{bmatrix} 
1 & 0 & 0\\\\\\ 
0 & 1 & 0\\\\\\ 
0 & 0 & 1 \end{bmatrix} = I_3
$$

\\(T\\) disparaît puisque \\(t_x\\), \\(t_y\\), \\(t_z \approx 0\\).

*Rappel de règles de dérivation : \\((\text{constant } a)' \rightarrow 0 \text{, } (\sin)' \rightarrow \cos \text{, } (\cos)' \rightarrow -\sin\\).*
{{< /alert >}}

$$\begin{align}
    \frac{\partial F^j}{\partial \alpha} &= N^j . \[I_3 . I_3 . \frac{\partial R\_\alpha}{\partial \alpha} | 0\] . 
    P\_o^j\\\\\\
    &= N^j . \frac{\partial \begin{bmatrix}1 & 0 & 0\\\\\\ 0 & \cos (\alpha \approx 0) & -\sin (\alpha \approx 0)\\\\\\ 0 & \sin (\alpha \approx 0) & \cos (\alpha \approx 0) \end{bmatrix}}{\partial \alpha} . \begin{bmatrix}X^j\\\\\\ Y^j\\\\\\ Z^j\end{bmatrix}\_o\\\\\\
    &= N^j . \begin{bmatrix}0 & 0 & 0\\\\\\ 0 & -\sin (\alpha \approx 0) & -\cos (\alpha \approx 0)\\\\\\ 0 & \cos (\alpha \approx 0) & -\sin (\alpha \approx 0) \end{bmatrix} . \begin{bmatrix}X^j\\\\\\ Y^j\\\\\\ Z^j\end{bmatrix}\_o\\\\\\
    &= N^j . \begin{bmatrix}0 & 0 & 0\\\\\\ 0 & 0 & -1\\\\\\ 0 & 1 & 0 \end{bmatrix} . \begin{bmatrix}X^j\\\\\\ Y^j\\\\\\ Z^j\end{bmatrix}\_o\\\\\\
    &= N^j . \begin{bmatrix}0\\\\\\ -Z^j\\\\\\ Y^j\end{bmatrix}\_o\\\\\\
\end{align}$$

Le raisonnement est le même pour \\(\frac{\partial F^j}{\partial \beta}, \frac{\partial F^j}{\partial \gamma}, \frac{\partial F^j}{\partial t\_x}, \frac{\partial F^j}{\partial t\_y}, \frac{\partial F^j}{\partial t\_z}\\) et on obtient :

$$
\frac{\partial F^j}{\partial \beta} = N^j . \begin{bmatrix}Z^j\\\\\\ 0\\\\\\ -X^j\end{bmatrix}\_o \text{, }
\frac{\partial F^j}{\partial \gamma} = N^j . \begin{bmatrix}-Y^j\\\\\\ X^j\\\\\\ 0\end{bmatrix}\_o
$$
$$
\frac{\partial F^j}{\partial t\_x} = N\_x^j \text{, }
\frac{\partial F^j}{\partial t\_y} = N\_y^j \text{, }
\frac{\partial F^j}{\partial t\_z} = N\_z^j
$$

Chacune de ces dérivées partielles est à implémenter dans la fonction `partial_derivatives` :

```python
def partial_derivatives(normal_vector, P_c):
    # ********************************************************************* #
    # A COMPLETER.                                                          #
    # Input:                                                                #
    #   normal_vector : ndarray[3] contenant la normale pour le segment     #
    #                   auquel appartient P_c                               #
    #   P_c : ndarray[3] le point de l'objet transformé dans Rc             #
    # Output:                                                               #
    #   partial_derivative : ndarray[6] derivee partielle du critere        #
    #               d'erreur pour chacun des parametres extrinseques        #
    #   crit_X0 : float64 - valeur du critere pour les parametres           #
    #               courants, qui servira de valeur initiale avant          #
    #               la mise a jour et le recalcul de l'erreur               #
    # ********************************************************************* #
    X, Y, Z = P_c[0], P_c[1], P_c[2]
    partial_derivative = np.zeros((6))
    
    # Variable a remplir
    partial_derivative[0] = 0
    partial_derivative[1] = 0
    partial_derivative[2] = 0
    partial_derivative[3] = 0
    partial_derivative[4] = 0
    partial_derivative[5] = 0

    # ********************************************************************
    crit_X0 = normal_vector[0] * X + normal_vector[1] * Y + normal_vector[2] * Z
    return partial_derivative, crit_X0
```

On peut ainsi formaliser le problème comme étant :

$$
F = J \Delta X \text{ avec } F = \begin{bmatrix}-F^1(X\_0) \\\\\\ \vdots \\\\\\ -F^{2n}(X\_0)\end{bmatrix}\_{2n\times 1} 
$$
$$
\text{ et } J = \begin{bmatrix}\frac{\partial F^1}{\partial \alpha} & \frac{\partial F^1}{\partial \beta} & \frac{\partial F^1}{\partial \gamma} & \frac{\partial F^1}{\partial t\_x} & \frac{\partial F^1}{\partial t\_y} & \frac{\partial F^1}{\partial t\_z}\\\\\\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\\\\\ \frac{\partial F^{2n}}{\partial \alpha} & \frac{\partial F^{2n}}{\partial \beta} & \frac{\partial F^{2n}}{\partial \gamma} & \frac{\partial F^{2n}}{\partial t\_x} & \frac{\partial F^{2n}}{\partial t\_y} & \frac{\partial F^{2n}}{\partial t\_z}\end{bmatrix}\_{2n\times 6}
$$
$$
\text{ et } \Delta X = \begin{bmatrix}\Delta \alpha \\\\\\ \vdots \\\\\\ \Delta t_z\end{bmatrix}\_{6 \times 1}
$$

>\\(2n\\) est le nombre de points sélectionnés, \\(n\\) est le nombre d'arêtes (une arête = deux extrémités).

\\(F\\) est connue, \\(J\\) est connue, il ne reste plus qu'à estimer \\(\Delta X\\) de sorte que l'égalité \\(F = J \Delta X\\) soit "la plus vraie possible". Pour cela, on souhaite minimiser la distance entre les deux côtés de l'égalité :

$$
\begin{align}
\min_{\Delta X} ||F - J\Delta X||^2 & \\\\\\
\Leftrightarrow \qquad \frac{\partial ||F - J\Delta X||^2}{\partial \Delta X} &= 0
\end{align}
$$

> En effet, si la dérivée de la fonction à minimiser est 0, alors la courbe de la fonction a bien atteint un minimum.

{{< alert type="info" >}}
*Rappel des opérations sur les matrices :*

\\((A+B)^T = A^T + B^T\\)

\\((AB)^T = B^T A^T\\)

\\(A\times B \neq B\times A\\)

\\(A^2 = A^T A\\)
{{< /alert >}}

En développant \\((F - J \Delta X)^2\\), on arrive à :

$$
\begin{align}
(F - J \Delta X)^2 & = (F - J \Delta X)^T(F - J \Delta X)\\\\\\
 &= (F^T - \Delta X^T J^T)(F - J \Delta X)\\\\\\
 &= F^TF - F^TJ\Delta X - \Delta X^T J^T F + \Delta X^T J^T J \Delta X
\end{align}
$$

On se sert des propriétés matricielles pour montrer que \\(F^TJ\Delta X = ((J\Delta X)^T F)^T = (\Delta X^T J^T F)^T\\). Considérons les tailles de matrices d'un côté et de l'autre de l'équation :
$$
\begin{align}
F^TJ\Delta X &\rightarrow [1\times2n][2n\times 6][6\times 1] \rightarrow [1\times 1]\\\\\\
(\Delta X^T J^T F)^T &\rightarrow [1\times6][6\times 2n][2n\times 1] \rightarrow [1\times 1]
\end{align}
$$

Etant donné que chacun de ces termes représente une matrice \\([1\times 1]\\), on peut écrire \\((\Delta X^T J^T F)^T = \Delta X^T J^T F\\), et par conséquent \\(F^TJ\Delta X = \Delta X^T J^T F\\).

On en déduit :
$$
\begin{align}
(F - J \Delta X)^2 & = F^TF - 2\Delta X^T J^T F + \Delta X^T J^T J \Delta X\\\\\\
\end{align}
$$

{{< alert type="info" >}}
*Quelques règles de dérivation :*

\\(\frac{\partial AX}{\partial X} = A^T\\), \\(\frac{\partial X^TA^T}{\partial X} = A^T\\), \\(\frac{\partial X^TAX}{\partial X} = 2AX\\).
{{< /alert >}}

$$
\begin{align}
\frac{\partial (F - J \Delta X)^2}{\partial \Delta X} & = -2J^T F + 2 J^T J \Delta X\\\\\\
&= -J^T F + J^T J \Delta X\\\\\\
\Leftrightarrow \qquad J^T F &= J^T J \Delta X \\\\\\
(J^TJ)^{-1} J^T F &= \Delta X \\\\\\
\end{align}
$$

**En résumé :** 

* minimiser la distance entre \\(F\\) et \\(J\Delta X\\) revient à dire que \\(\frac{\partial (F - J \Delta X)^2}{\partial \Delta X} = 0\\) 
* la solution est \\(\Delta X = (J^TJ)^{-1} J^T F\\). 

>La matrice \\(J^+ = (J^TJ)^{-1} J^T\\) s'appelle la *pseudo-inverse* de \\(J\\).

Dans le code Python, on peut ainsi implémenter la mise à jour des paramètres dans la boucle d'optimisation. \\(\Delta X\\) correspond à la variable nommée `delta_solution`. On peut ensuite passer `delta_solution` à la fonction `matTools.construct_matrix_from_vec` qui renvoie une matrice des incréments de \\(X\\) de la même forme que `extrinsic` (la matrice des paramètres extrinsèques).

On incrémente chacun des éléments de `extrinsic` en la multipliant par `delta_extrinsic`.

```python
# ********************************************************************* #
# A COMPLETER.                                                          #
# delta_solution = ...                                                  #
# delta_extrinsic = matTools.construct_matrix_from_vec(delta_solution)  #
# extrinsic = ...                                                       #
# ********************************************************************* #
```

Une fois la boucle d'optimisation opérationnelle, on peut visualiser la transformation du modèle sans la boite virtuelle en enlevant les 12 premiers points de `model3D_Ro` :

{{< img src="images/final_objective_without_box.png" align="center" title="Transformation du modèle après estimation de la pose caméra" >}}
{{< vs 1 >}}

## ***Etape 4*** : projection de la pose estimée sur l'image prise d'un point de vue différent {#anchor-step-4}

Une deuxième photo a été capturée d'un point de vue différent, et la matrice de passage entre le premier et le deuxième point de vue est stockée et chargée en début de programme depuis le fichier `.txt` correspondant. Elle permet de re-projeter le modèle dans l'image issue de la deuxième caméra et d'en afficher le résultat :

```python
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)
ax5.set_xlim(0,720)
ax5.set_ylim(480)
plt.imshow(image_2)

# A completer avec la matrice de passage du repere Ro vers Rc2, avec les matrices 
# Ro -> Rc et Rc -> Rc2
Ro_to_Rc2 = ...

transform_and_draw_model(model3D_Ro[12:], intrinsic_matrix, Ro_to_Rc2, ax5)
plt.show(block = False)
```

{{< img src="images/left_to_right.png" align="center" title="Projection dans la deuxième image" >}}
{{< vs 1 >}}

## Bonus pour la fin {#bonus}

Grâce à la matrice des paramètres extrinsèques estimée, et tant que l'origine du repère objet ne change pas, on peut ajouter des éléments à la scène 3D et les projeter dans l'image de manière identique. `model3D_Ro_final` contient les  points d'une scène contenant Pikachu et un dinosaure.
 
{{< img src="images/pika_dino_blender.png" align="center" title="Ajout d'élements à la scène 3D" >}}
{{< vs 1 >}}

On décommente les dernières lignes d'affichage pour obtenir le résultat final :

```python
fig6 = plt.figure(6)
ax6, lines = utils.plot_3d_model(model3D_Ro_final, fig6)

fig7 = plt.figure(7)
ax7 = fig7.add_subplot(111)
ax7.set_xlim(0, 720)
plt.imshow(image)
transform_and_draw_model(model3D_Ro_final[12:], intrinsic_matrix, extrinsic_matrix, ax7)
plt.show(block=True)
```

{{< img src="images/pika_dino.png" align="center" title="Nouvelle scène projetée" >}}
{{< vs 1 >}}

TP crée par notre proffesseur [Claire Labit-Bonis](https://clairelabitbonis.github.io/).
