---
title: "3DP-TP-01 | Monocular localization with iterative PnL"
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
    name: Mono. localization
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


>This exercise consists in filling empty functions, or parts of the `localisation.py` file:
>- [step 1](#anchor-step-1) : `perspective_projection` and `transform_and_draw_model`,
>- [step 2](#anchor-step-2) : `calculate_normal_vector` and `calculate_error`
>- [step 3](#anchor-step-3) : `partial_derivatives`,
>- [step 4](#anchor-step-4) : transformation towards a second point of view.
>
>This post describes the role of each function.

{{< alert type="danger" >}}
Do not dive headfirst into the code ! The first part "goal description" is just a global and theoretical presentation of the subject. Each step and its associated functions are described in details within their respective parts: [step 1](#anchor-step-1), [step 2](#anchor-step-2), [step 3](#anchor-step-3), [step 4](#anchor-step-4).
{{< /alert >}}

## Contents download
Practical session files can be downloaded through your course on Moodle, or at [this link](files/files.zip). 

## Goal description {#anchor-step-0}
This exercise aims at finding the optimal transformation between a 3D model expressed in centimeters in an *object* coordinate system \\(\mathcal{R\_o} (X,Y,Z)\\) (sometimes called *world* coordinate system \\(\mathcal{R\_w}\\)) in order to draw it as an overlay in a 2D pixel image defined in its *image* coordinate system \\(\mathcal{R\_i} (u,v)\\). The 3D model was downloaded from [free3d](https://free3d.com) and modified within [Blender](https://www.blender.org/) for the needs of this exercise. Points and edges were respectively exported *via* a Python script to the `pikachu.xyz` and `pikachu.edges` files.


{{< img src="images/blender_pikachu.png" align="center" title="3D model made in Blender" >}}
{{< vs 1 >}}

`pikachu.xyz` contains 134 lines and 3 columns, corresponding to the 134 model points and their three \\(x, y, z\\) coordinates. 

`pikachu.edges` contains 246 lines and 2 columns, corresponding to the 246 model edges and their 2 ends indices (for instance, the first line in `pikachu.edges` describes the first edge: its first point coordinates are at index 2 in the `pikachu.xyz` file, and its second point is at index 0).

{{< img src="images/edges_xyz.png" align="center" title="Coordinates and edges files" >}}
{{< vs 1 >}}

By "transformation", we mean scene rotation and translation along the \\(x\\), \\(y\\) and \\(z\\) axes for each point in \\(\mathcal{R\_o}\\).

In our case, we want to perform augmented reality by positioning Pikachu's model over the sky-blue cube of the image below, and making it perfectly match with the virtual cube on which it is standing.

{{< img src="images/final_objective.png" align="center" title="Goal to be reached" >}}
{{< vs 1 >}}

<details>
<summary><strong>Intrinsic parameters. <span style="color:pink">Click to expand</span></strong></summary> 

We use the pinhole camera model allowing to perform this operation through two successive transformations: \\(\mathcal{R\_o} \rightarrow \mathcal{R\_c}\\) and \\(\mathcal{R\_c} \rightarrow \mathcal{R\_i}\\). Apart from \\(\mathcal{R\_o}\\) and \\(\mathcal{R\_i}\\) coordinate systems, we then must also considerate the camera frame \\(\mathcal{R\_c}\\). 

>The resource *[Modélisation et calibrage d'une caméra](http://www.optique-ingenieur.org/fr/cours/OPI_fr_M04_C01/co/Grain_OPI_fr_M04_C01_2.html)* describes the pinhole camera model in details, and can help understanding the exercise.

{{< img src="images/goal.png" align="center" title="Coordinate systems transformation" >}}
{{< vs 1 >}}

>Wikipedia defines the [pinhole model](https://en.wikipedia.org/wiki/Pinhole_camera_model) as describing *the mathematical relationship between the coordinates of a point in three-dimensional space and its projection onto the image plane of an ideal pinhole camera, where the camera aperture is described as a point and no lenses are used to focus light*. 

The change between \\(\mathcal{R\_c}\\) and \\(\mathcal{R\_i}\\) coordinate systems is done thanks to the **intrinsic** camera parameters. These \\((\alpha\_u, \alpha\_v, u\_0, v\_0)\\) coefficients are then stored in a homogeneous transformation matrix \\(K\_{i \leftarrow c}\\) so that we can describe the relation \\(p\_i = K\_{i \leftarrow c}.P\_c\\) as:

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

with:

- \\(p\_i\\) (on the left) the pixel point within the 2D image frame \\(\mathcal{R\_i}\\),
- \\(P\_c\\) (on the right) the point in centimeters within the 3D camera frame \\(\mathcal{R\_c}\\),
- \\(s = \frac{1}{Z}\\),
- \\(\alpha_u = k_x f\\), \\(\alpha_v = k_y f\\):
    - \\(k_x = k_y\\) the sensor number of pixels per millimeter along \\(x\\) and \\(y\\) directions -- the equality being true only if the pixels are square,
    - \\(f\\) the focal distance.
- \\(u_0\\) and \\(v_0\\) the image centers in pixels within \\(\mathcal{R}_i\\).

In our case, the image was captured by a Canon EOS 700D sensor of size \\(22.3\times14.9 mm\\). The image size being \\(720\times480 px\\) and the focal distance \\(18mm\\), we deduce the parameters \\(\alpha\_u = 581.1659\\), \\(\alpha\_v = 579.8657\\), \\(u\_0 = 360\\) and \\(v\_0 = 240\\) stored in the `calibration_parameters.txt` file.
</details>

<details>
<summary><strong>Extrinsic parameters. <span style="color:pink">Click to expand</span></strong></summary> 

To reach our goal *i.e.*, display our 3D model in the 2D image in pixels, we have to estimate the \\(\mathcal{R\_o}\rightarrow\mathcal{R\_c}\\) transformation coefficients and establish the relation \\(P\_c=M\_{c\leftarrow o}.P\_o\\), with:

- \\(M\_{c\leftarrow o}=\big[ R\_{\alpha\beta\gamma} | T \big]\\) the homogeneous transformation matrix of the Euler angles and \\(x\\), \\(y\\) and \\(z\\) translation of the coordinate system.

- \\(R\_{\alpha\beta\gamma}\\) the rotation matrix resulting from the successive applications -- and thus the multiplication between them -- of the rotation matrices \\(R\_\gamma\\), \\(R\_\beta\\) and \\(R\_\alpha\\):

$$R\_\alpha = \begin{bmatrix}1 & 0 & 0\\\\0 & \cos \alpha & -\sin \alpha\\\\0 & \sin \alpha & \cos \alpha\end{bmatrix}$$

$$R\_\beta = \begin{bmatrix}\cos \beta & 0 & -\sin \beta\\\\0 & 1 & 0\\\\\sin \beta & 0 & \cos \beta\end{bmatrix}$$

$$R\_\gamma = \begin{bmatrix}\cos \gamma & -\sin \gamma & 0\\\\\sin \gamma & \cos \gamma & 0\\\\0 & 0 & 1\end{bmatrix}$$

With the correct extrinsic parameters \\((\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\), the 3D model in its coordinate system \\(\mathcal{R\_o}\\) can be transformed into \\(\mathcal{R\_c}\\) then \\(\mathcal{R\_i}\\), while respecting the following relationship for each point \\(P_o\\):

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
The \\(\bigodot\\) operator is the multiplication between terms after having removed the homogeneous coordinate from \\(M\_{c\leftarrow o} P\_o\\). It is only here so that matrices shapes match.
{{< /alert >}}
</details>

</br>

>The instrinsic parameters are known; in order to find the extrinsic parameters, we use a localization method (*i.e.* of parameters estimation) called **Perspective-n-Lines**.

### Extrinsic parameters estimation

Our starting point is an initial coarse estimate of \\((\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\). This initial estimate is more or less accurate depending on our *a priori* knowledge of the scene, the object and the position of the camera.

In our case, the initial parameters have values \\(alpha = -2.1\\), \\(beta = 0.7\\), \\(gamma = 2.7\\), \\(t\_x = 3.1\\), \\(t\_y = 1.3\\) and \\(t\_z = 18\\), *i.e.,* we know before we even start that the 3D model will have to undergo a rotation of angles \\((-2.1, 0.7, 2.7)\\) around its three axes, and that it should shift approximately \\(3cm\\) in \\(x\\), \\(1.5cm\\) in \\(y\\) and \\(18cm\\) in \\(z\\). 

>This *a priori* knowledge of the environment comes from the fact that one is able, as a human being, to evaluate the distance of the real object from the sensor only by looking at the image. 

### Use of the programme

When the script `localisation.py` is launched, two figures open: one represents a real scene captured by the camera, the other represents the virtual model to be transformed and whose system origin is materialised by a red dot. As the objective is to copy the box on which the Pikachu rests on top of the sky-blue cube on the table, you must first select five edges belonging to the real box (by *clicking* the mouse on the edges ends), then select the corresponding edges in the same order on the 3D model (by *clicking* the mouse in the middle of the edges). The number of segments to be selected (by default 5), is fixed from the start in the variable `nb_segments`.

{{< img src="images/edge_selection.png" align="center" title="Edge selection in the image and in the 3D model" >}}
{{< vs 1 >}}

In this way, it will be possible to calculate the error on the estimation of extrinsic parameters by comparing the distance between the edges selected in the image and those of the transformed model. This error criterion is described at [step 2](#anchor-step-2).

## ***Step 1***: Display the pattern in \\(\mathcal{R_i}\\) {#anchor-step-1}

In the main program, the points of the model are stored in `model3d_Ro`, a matrix of size \\([246\times6]\\) corresponding to the 246 edges of the model, each defined by the 6 coordinates of its two points \\(P_1(x_1, y_1, z_1)\\) and \\(P_2(x_2, y_2, z_2)\\). The transformation matrix \\(M\_{c\leftarrow o}\\) is stored in `extrinsic_matrix` and the intrinsic parameters of the camera are stored in `intrinsic_matrix`.

To visualise the 3D model in the 2D image and get an idea of the accuracy of our estimate, we have to code the `tranform_and_draw_model` function which allows to apply the \\(K\_{i \leftarrow c} M\_{c\leftarrow o}\\) transformation at each point \\(P\_o\\) of a set of edges `edges_Ro`, with an `intrinsic` matrix, and an `extrinsic` matrix to finally display the result in a `fig_axis` figure:

``` python
def transform_and_draw_model(edges_Ro, intrinsic, extrinsic, fig_axis):
    # ********************************************************************* #
    # TO BE COMPLETED.                                                      #
    # FUNCTIONS YOU WILL HAVE TO USE :                                      #
    #   - perspective_projection                                            #  
    #   - transform_point_with_matrix                                       #
    # Input:                                                                #
    #   edges_Ro : ndarray[Nx6]                                             #
    #             N = number of edges in the model                          #
    #             6 = (X1, Y1, Z1, X2, Y2, Z2) the P1 and P2 point          #
    #                 coordinates for each edge                             #
    #   intrinsic : ndarray[3x3] - camera intrinsic parameters              #
    #   extrinsic : ndarray[4x4] - camera extrinsic parameters              #
    #   fig_axis : figure used for display                                  #
    # Output:                                                               #
    #   No return - the function only calculates and displays the           #
    #   transformed points (u1, v1) and (u2, v2)                            #
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

The objective is to store in the points `[u_1, v_1]` and `[u_2, v_2]` the coordinates \\((u, v)\\) of the points \\(P_1\\) and \\(P_2\\) of the edges after transformation.

This function can be divided into two sub-steps: 

* the transformation \\(\mathcal{R\_o} \rightarrow \mathcal{R\_c}\\) of the points \\(P\_o\\) using the function `transform_point_with_matrix` provided in the `matTools`  library:

    ```python
    P_c = matTools.transform_point_with_matrix(extrinsic, P_o)
    ```
        
* the projection \\(\mathcal{R\_c} \rightarrow \mathcal{R\_i}\\) of the newly obtained \\(P\_c\\) points using the function `perspective_projection` which must be completed :
    
    ```python
    def perspective_projection(intrinsic, P_c):
        # ***************************************************** #
        # TO BE COMPLETED.                                      #
        # Useful function:                                      #
        #   np.dot                                              #
        # Input:                                                #
        #   intrinsic : ndarray[3x3] - intrinsic parameters     #
        #   P_c : ndarray[Nx3],                                 #
        #         N = number of points to transform             #
        #         3 = (X, Y, Z) the points coordinates          #
        # Output:                                               #
        #   u, v : two ndarray[N] containing the Ri coordinates #
        #          of the transformed Pc points                 #
        # ***************************************************** #
        
        u, v = 0, 0 # Part to replace
        
        return u, v
    ```

Once `perspective_projection` and `transform_and_draw_model` have been completed, the launch of the overall programme will project the model in the image, with the extrinsic parameters defined at the outset.

{{< img src="images/first_rough_estimate.png" align="center" title="First projection of the model in the image" >}}
{{< vs 1 >}}

>The transformation applied to these points is obviously not very good, the model does not match the box as one would wish. In order to be able to differentiate a "bad" estimate of the extrinsic parameters from a "good" one, and thus be able to automatically propose a new estimate closer to our objective, we must determine an error criterion characterising the distance we are from this optimal objective.

## ***Step 2***: Determine an error criterion {#anchor-step-2}

The error criterion is used to assess the accuracy of our estimate. The greater the error, the poorer our extrinsic parameters. Once we have determined this error criterion, we can integrate it into an optimisation loop aimed at minimising it, and thus have the best possible extrinsic parameters for our 2D/3D matching objective.

As an example, the figure below illustrates this optimisation loop for the projection of the virtual box on the real cube. At iteration \\(0\\), the parameters are coarse, the error is large. By changing the parameters the error will be reduced until ideally zero error and optimal transformation for a perfect projection of the 3D model into the 2D image is achieved.

{{< img src="images/extrinsic_optim.gif" align="center" title="Optimisation of extrinsic parameters" >}}
{{< vs 1 >}}

In 2D/3D segment matching, the error criterion to be minimised is the scalar product of the normal to the interpretation plane for the matching. In other words, the objective is to transform the points of the model so that the segments selected in the image and those selected in the model belong to the same plane expressed in the camera frame \\(\mathcal{R\_c}\\).

{{< img src="images/interpretation_plan.png" align="center" title="Interpretation plan relating to line matching" >}}
{{< vs 1 >}}

As shown in the figure above, the *interpretation plane* can be defined as being formed by the segments \\(\mathcal{l\_{i \rightarrow c}^{j,1}}\\) and \\(\mathcal{l\_{i \rightarrow c}^{j,2}}\\). In this notation, \\(j\\) corresponds to the number of edges selected when the program is launched (5 by default). For each selected edge \\(j\\) and expressed in the *image* frame \\(\mathcal{R_i}\\), there are two segments \\(l^1\\) and \\(l^2\\). The \\(l^1\\) segment is composed of the two ends \\(P\_{i \rightarrow c}^0\\) and \\(P\_{i \rightarrow c}^1\\); the \\(l^2\\) segment is composed of the two ends \\(P\_{i \rightarrow c}^1\\) and \\(P\_{i \rightarrow c}^2\\). Each of the \\(P\_{i \rightarrow c}\\) is a point in the *image* frame \\(\mathcal{R\_i}\\) transformed into the *camera* frame \\(\mathcal{R\_c}\\). They are known: they are the ones that have been selected by mouse click.

Once these segments have been calculated, we can calculate the normal to the interpretation plane \\(N\_c^j\\). As a reminder:

$$N = \frac{l^1 \wedge l^2}{||l^1 \wedge l^2||}$$

In the segment selection function `utils.select_segments()`, as you make edge selections, the normals are calculated and stored in the `normal_vectors` matrix thanks to the `calculate_normal_vector` function which must be completed in the `localisation.py` file:

```python
def calculate_normal_vector(p1_Ri, p2_Ri, intrinsic):
    # ********************************************************* #
    # TO BE COMPLETED.                                          #
    # Useful functions:                                         #
    #   np.dot, np.cross, np.linalg.norm, np.linalg.inv         #
    # Input:                                                    #
    #   p1_Ri : list[3]                                         #
    #           3 = (u, v, 1) of the first selected point       #
    #   p2_Ri : list[3]                                         #
    #           3 = (u, v, 1) of the second selected point      #
    #   intrinsic : ndarray[3x3] of intrinsic                   #
    # Output:                                                   #
    #   normal_vector : ndarray[3] containing the normal to     #
    #                   L1_c and L2_c segments, deducted from   #
    #                   the selected image points               #
    # ********************************************************* #

    normal_vector = np.zeros((len(p1_Ri),)) # Part to replace

    return normal_vector
```

Then, for each segment \\(j \in \[1, ..., 5\]\\), the distance between the plane linked to the image segments and the points of the 3D model can be calculated using the scalar product of the \\(N\_c^j\\) and the \\(P\_{o\rightarrow c}^j\\). If the scalar product is null, both vectors are orthogonal and the point \\(P\_{o\rightarrow c}^j\\) belongs to the same plane as \\(P\_{i \rightarrow c}^j\\). The larger the scalar product, the further the extrinsic parameters move away from the correct transformation.

To summarize, the parameter optimization criterion is \\(\sum\_{j=1}^{2n} F^j(X)^2\\) (the square removes negative values), with \\(F^j(X) = N^{j \; \text{mod} \; 2}.P\_
{o\rightarrow c}^{j \; \text{mod} \; 2, 1|2}\\), depending on whether one is on the point \\(P^{1}\\) or \\(P^{2}\\). 

> In the sum that runs through the points from \\(j\\) to \\(2n\\), \\(j \; \text{mod} \; 2\\) is the segment index for the current point. In other words, we cumulate the \\((N^i.P^{i, 1})^2\\) and \\((N^i.P^{i, 2})^2\\) for all \\(i\\) segments.

{{< alert type="warning" >}}
Here, \\(X\\) designates the set of parameters \\((\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\) and not a coordinate.
{{< /alert >}}

Remember that every \\(P\_{o\rightarrow c} = \big[ R\_{\alpha\beta\gamma} | T \big]. P\_o\\) ; the value of the error thus actually depends on the value of the extrinsic parameters. Each time the optimization loop is run, changing the weights of these parameters will influence the \\(F(X)\\) criterion.

The function `calculate_error` calculates the error criterion. It takes as input the number of selected edges `nb_segments`, the `normal_vectors`, and the `Rc_segments` selected and then transformed by the matrix of extrinsic parameters and expressed in \\(\mathcal{R_c}\\).

```python
def calculate_error(nb_segments, normal_vectors, segments_Rc):
    # ***************************************************************** #
    # TO BE COMPLETED.                                                  #
    # Input:                                                            #
    #   nb_segments : default 5 = number of selected segments           #
    #   normal_vectors : ndarray[Nx3] - normal vectors to the           #
    #                    interpretation plane of selected segments      #
    #                    N = number of segments                         #
    #                    3 = (X,Y,Z) normal coordinates in Rc           #
    #   segments_Rc : ndarray[Nx6] = selected segments in Ro            #
    #                 and transformed in Rc                             #
    #                 N = number of segments                            #
    #                 6 = (X1, Y1, Z1, X2, Y2, Z2) of points P1 and     #
    #                 P2 of the transformed edges in Rc                 #
    # Output:                                                           #
    #   err : float64 - cumulated error of observed vs. expected dist.  #
    # ***************************************************************** #
    err = 0
    for p in range(nb_segments):
        # Part to replace with the error calculation
        err = err + 0
    
    err = np.sqrt(err / 2 * nb_segments)
    return err
```

> From a mathematical point of view, the sum is written for \\(j\\) ranging from \\(1\\) to \\(2n\\), *i.e.,* the number of points. From an implementation point of view and given the construction of our variables, it is simpler to express the criterion through a sum running along the edges: \\(\sum\_{j=1}^{n} F^j(X)\\) with \\(F^{j}(X) = F^{j, 1}(X)^2 + F^{j, 2}(X)^2\\).
>
> Thus, \\(F^{j, 1}(X) = N^j.P\_{o\rightarrow c}^{j, 1}\\) and \\(F^{j, 2}(X) = N^j.P\_{o\rightarrow c}^{j, 2}\\). 

## ***Step 3***: Estimation of parameters by ordinary least squares method {#anchor-step-3}

<!-- [//]: # (At each step, we look for an increment to apply to each of the parameters \\(X (\alpha, beta, gamma, t\_x, t\_y, t\_z)$. For the example, we can focus our attention on the parameter \\(\gamma\\) of rotation around the axis \\(Z$. The procedure can then be illustrated by the following figure: ![Successive increments of extrinsic parameters](images/increment_gamma.png) The starting point is expressed in its object reference system \\(\mathcal{R\_O}$. At the iteration \\(k=0\\), the transformation performed is: \\($\begin{bmatrix}X \\\\\ Y \\\end{bmatrix}\_{\mathcal{R\_C\_0}} = \[R\_\gamma . R\_\beta . R\_\alpha | T\_{xyz}\]\_{k=0}\begin{bmatrix}X \\\\\\ Y \\\\ Z\end{bmatrix}\_{\mathcal{R\_{O}}}$\\) If we call \\(M\_{k=0}\\) the matrix of extrinsic parameters \\(\[R\_\gamma .  R\_{\Delta\beta}. R\_{\Delta\alpha} | T\_{\Delta x, \Delta y, \Delta z}\]\_{k=1}. M\_{k=0}.\begin{bmatrix} X \\\\Delta Y \Delta Z \end{bmatrix}\_{\mathcal{R\_{O}}}$$) -->

At each iteration \\(k\\), we look for a set of parameters \\(X\_{k}(\alpha, \beta, \gamma, t\_x, t\_y, t\_z)\\) such as the criterion \\(F(X)\\) is equal to the criterion for the previous parameter set \\(X_0\\) (before update) incremented by a delta weighted by the Jacobian of the function. 

We thus seek to solve a system of the form :

$$F(X) \approx F(X\_0) + J \Delta X$$

>The Jacobian \\(J\\) contains on its lines the partial derivatives of the \\(F\\) function for each selected point and according to each of the \\(X\\) parameters. It reflects the trend of the criterion (ascending/descending) and the speed at which it increases or decreases according to the value of each of its parameters.

Since our objective is to reach a criterion \\(F(X) = 0\\), we translate the problem to be solved by:

$$
\begin{align}
0 &= F(X\_0) + J \Delta X \\\\\\
\Leftrightarrow \qquad -F(X\_0) &= J\Delta X
\end{align}
$$

The advantage of working with successive increments is that the increment values are so small in relation to the last iteration that the variation of these parameters can be approximated as 0. As a reminder, the error criterion is expressed as follows for each segment \\(j\\):

$$
\begin{align}
F^{j, 1}(X) &= N^j.P\_{o\rightarrow c}^{j, 1} \text{ with } P\_{o\rightarrow c}^{j, 1} = \big[ R\_{\alpha\beta\gamma
} | T \big] . P\_o^{j, 1}\\\\\\
F^{j, 2}(X) &= N^{j}.P\_{o\rightarrow c}^{j, 2} \text{ with } P\_{o\rightarrow c}^{j, 2} = \big[ R\_{\alpha\beta\gamma
} | T \big] . P\_o^{j, 2}
\end{align}
$$

Because of always having \\(\Delta X \approx 0\\), the calculation of the Jacobian matrix is considerably simplified, since the partial derivatives \\((\frac{\partial F^j}{\partial \alpha}, \frac{\partial F^j}{\partial \beta}, \frac{\partial F^j}{\partial \gamma}, \frac{\partial F^j}{\partial t\_x}, \frac{\partial F^j}{\partial t\_y}, \frac{\partial F^j}{\partial t\_z})\\) are the same at each iteration.

The detail of the derivation is given for the first parameter \\(\alpha\\), the following are to be demonstrated :

$$\begin{align}
    \frac{\partial F^j}{\partial \alpha} &= N^j . [R\_\gamma . R\_\beta . \frac{\partial R\_\alpha}{\partial \alpha} | T] . P\_o^j\\\\\\
\end{align}
$$

{{< alert type="info" >}}
\\(R\_\gamma\\) and \\(R\_\beta\\) disappear from the derivation because for \\(\gamma \approx 0\\) and \\(\beta \approx 0\\), we have:

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

\\(T\\) disappears since \\(t_x\\), \\(t_y\\), \\(t_z \approx 0\\).

*Reminder of derivation rules: \\((\text{constant } a)' \rightarrow 0 \text{, } (\sin)' \rightarrow \cos \text{, } (\cos)' \rightarrow -\sin\\).*
{{< /alert >}}

$$
\begin{align}
    \frac{\partial F^j}{\partial \alpha} &= N^j . \[I_3 . I_3 . \frac{\partial R\_\alpha}{\partial \alpha} | 0\] . 
    P\_o^j\\\\\\
    &= N^j . \frac{\partial \begin{bmatrix}1 & 0 & 0\\\\\\ 0 & \cos (\alpha \approx 0) & -\sin (\alpha \approx 0)\\\\\\ 0 & \sin (\alpha \approx 0) & \cos (\alpha \approx 0) \end{bmatrix}}{\partial \alpha} . \begin{bmatrix}X^j\\\\\\ Y^j\\\\\\ Z^j\end{bmatrix}\_o\\\\\\
    &= N^j . \begin{bmatrix}0 & 0 & 0\\\\\\ 0 & -\sin (\alpha \approx 0) & -\cos (\alpha \approx 0)\\\\\\ 0 & \cos (\alpha \approx 0) & -\sin (\alpha \approx 0) \end{bmatrix} . \begin{bmatrix}X^j\\\\\\ Y^j\\\\\\ Z^j\end{bmatrix}\_o\\\\\\
    &= N^j . \begin{bmatrix}0 & 0 & 0\\\\\\ 0 & 0 & -1\\\\\\ 0 & 1 & 0 \end{bmatrix} . \begin{bmatrix}X^j\\\\\\ Y^j\\\\\\ Z^j\end{bmatrix}\_o\\\\\\
    &= N^j . \begin{bmatrix}0\\\\\\ -Z^j\\\\\\ Y^j\end{bmatrix}\_o\\\\\\
\end{align}
$$

The reasoning is the same for \\(\frac{\partial F^j}{\partial \beta}, \frac{\partial F^j}{\partial \gamma}, \frac{\partial F^j}{\partial t\_x}, \frac{\partial F^j}{\partial t\_y}, \frac{\partial F^j}{\partial t\_z}\\) and we get:

$$
\frac{\partial F^j}{\partial \beta} = N^j . \begin{bmatrix}Z^j\\\\\\ 0\\\\\\ -X^j\end{bmatrix}\_o \text{, }
\frac{\partial F^j}{\partial \gamma} = N^j . \begin{bmatrix}-Y^j\\\\\\ X^j\\\\\\ 0\end{bmatrix}\_o
$$
$$
\frac{\partial F^j}{\partial t\_x} = N\_x^j \text{, }
\frac{\partial F^j}{\partial t\_y} = N\_y^j \text{, }
\frac{\partial F^j}{\partial t\_z} = N\_z^j
$$

Each of these partial derivatives is to be implemented in the `partial_derivatives` function:

```python
def partial_derivatives(normal_vector, P_c):
    # ********************************************************************* #
    # TO BE COMPLETED.                                                      #
    # Input:                                                                #
    #   normal_vector : ndarray[3] contains the normal of the segment       #
    #                   to which P_c belongs                                #
    #   P_c : ndarray[3] the object point transformed to Rc                 #
    # Output:                                                               #
    #   partial_derivative : ndarray[6] partial derivative of the criterion #
    #                        for each extrinsic parameter                   #
    #   crit_X0 : float64 - criterion value for the current parameters,     #
    #                       which will be used as initial value before      #
    #                       the update and recalculation of the error       #
    # ********************************************************************* #
    X, Y, Z = P_c[0], P_c[1], P_c[2]
    partial_derivative = np.zeros((6))
    
    # Part to replace #######
    partial_derivative[0] = 0
    partial_derivative[1] = 0
    partial_derivative[2] = 0
    partial_derivative[3] = 0
    partial_derivative[4] = 0
    partial_derivative[5] = 0
    #########################

    # ********************************************************************
    crit_X0 = normal_vector[0] * X + normal_vector[1] * Y + normal_vector[2] * Z
    return partial_derivative, crit_X0
```

The problem can thus be formalised as:

$$
F = J \Delta X \text{ with } F = \begin{bmatrix}-F^1(X\_0) \\\\\\ \vdots \\\\\\ -F^{2n}(X\_0)\end{bmatrix}\_{2n\times 1} 
$$
$$
\text{ and } J = \begin{bmatrix}\frac{\partial F^1}{\partial \alpha} & \frac{\partial F^1}{\partial \beta} & \frac{\partial F^1}{\partial \gamma} & \frac{\partial F^1}{\partial t\_x} & \frac{\partial F^1}{\partial t\_y} & \frac{\partial F^1}{\partial t\_z}\\\\\\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\\\\\ \frac{\partial F^{2n}}{\partial \alpha} & \frac{\partial F^{2n}}{\partial \beta} & \frac{\partial F^{2n}}{\partial \gamma} & \frac{\partial F^{2n}}{\partial t\_x} & \frac{\partial F^{2n}}{\partial t\_y} & \frac{\partial F^{2n}}{\partial t\_z}\end{bmatrix}\_{2n\times 6}
$$
$$
\text{ and } \Delta X = \begin{bmatrix}\Delta \alpha \\\\\\ \vdots \\\\\\ \Delta t_z\end{bmatrix}\_{6 \times 1}
$$

>\\(2n\\) is the number of selected points, \\(n\\) is the number of edges (one edge = two ends).

\\(F\\) is known, \\(J\\) is known, all that remains is to estimate \\(\Delta X\\) so that the equality \\(F = J \Delta X\\) is "as true as possible". This is done by minimising the distance between the two sides of equality:

$$
\begin{align}
\min_{\Delta X} ||F - J\Delta X||^2 & \\\\\\
\Leftrightarrow \qquad \frac{\partial ||F - J\Delta X||^2}{\partial \Delta X} &= 0
\end{align}
$$

> Indeed, if the derivative of the function to be minimized is 0, then the curve of the function has definitely reached a minimum.

{{< alert type="info" >}}
*Reminder of matrices operations:*

\\((A+B)^T = A^T + B^T\\)

\\((AB)^T = B^T A^T\\)

\\(A\times B \neq B\times A\\)

\\(A^2 = A^T A\\)
{{< /alert >}}

By developing \\((F - J \Delta X)^2\\), we arrive at :

$$
\begin{align}
(F - J \Delta X)^2 & = (F - J \Delta X)^T(F - J \Delta X)\\\\\\
 &= (F^T - \Delta X^T J^T)(F - J \Delta X)\\\\\\
 &= F^TF - F^TJ\Delta X - \Delta X^T J^T F + \Delta X^T J^T J \Delta X
\end{align}
$$

We use matrices properties to show \\(F^TJ\Delta X = ((J\Delta X)^T F)^T = (\Delta X^T J^T F)^T\\). Let's consider matrices shapes for both sides of this equality:
$$
\begin{align}
F^TJ\Delta X &\rightarrow [1\times2n][2n\times 6][6\times 1] \rightarrow [1\times 1]\\\\\\
(\Delta X^T J^T F)^T &\rightarrow [1\times6][6\times 2n][2n\times 1] \rightarrow [1\times 1]
\end{align}
$$

Given that each of these terms is a \\([1\times 1]\\) matrix in the end, we can write \\((\Delta X^T J^T F)^T = \Delta X^T J^T F\\), and thus \\(F^TJ\Delta X = \Delta X^T J^T F\\).

We deduce from this:
$$
\begin{align}
(F - J \Delta X)^2 & = F^TF - 2\Delta X^T J^T F + \Delta X^T J^T J \Delta X\\\\\\
\end{align}
$$

{{< alert type="info" >}}
*Some rules of derivation:*

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

**In a nutshell:** 

* minimizing the distance between \\(F\\) and \\(J\Delta X\\) means that \\(\frac{\partial (F - J \Delta X)^2}{\partial \Delta X} = 0\\) 
* the solution is \\(\Delta X = (J^TJ)^{-1} J^T F\\). 

>The matrix \\(J^+ = (J^TJ)^{-1} J^T\\) is called the *pseudo-inverse* of \\(J\\).


In the Python code, we can thus implement the parameter update in the optimization loop. \\(\Delta X\\) corresponds to the variable named `delta_solution`. We can then pass `delta_solution` to the function `matTools.construct_matrix_from_vec` which returns a matrix of \\(X\\) increments in the same form as `extrinsic` (the extrinsic parameter matrix).


Each element of `extrinsic` is incremented by multiplying it by `delta_extrinsic`.

```python
# ********************************************************************* #
# TO BE COMPLETED.                                                      #
# delta_solution = ...                                                  #
# delta_extrinsic = matTools.construct_matrix_from_vec(delta_solution)  #
# extrinsic = ...                                                       #
# ********************************************************************* #
```

Once the optimisation loop is operational, the model transformation can be visualised without the virtual box by removing the first 12 points of `model3D_Ro` :

{{< img src="images/final_objective_without_box.png" align="center" title="Transformation of the model after estimation of the camera pose" >}}
{{< vs 1 >}}

## ***Step 4***: projection of the estimated pose on the image taken from a different point of view {#anchor-step-4}

A second photo has been captured from a different point of view, and the passage matrix between the first and second views is stored and loaded at the beginning of the programme from the corresponding `.txt` file. This allows the model to be re-projected into the image from the second camera, and the result to be displayed:

```python
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)
ax5.set_xlim(0.720)
ax5.set_ylim(480)
plt.imshow(image_2)

# To be completed with the passage matrix from Ro to Rc2
# Ro -> Rc and Rc -> Rc2
Ro_to_Rc2 = ...

transform_and_draw_model(model3D_Ro[12:], intrinsic_matrix, Ro_to_Rc2, ax5)
plt.show(block = False)
```

{{< img src="images/left_to_right.png" align="center" title="Projection in the second image" >}}
{{< vs 1 >}}

## Bonus for the end {#bonus}

Thanks to the estimated extrinsic parameter matrix, and as long as the origin of the object reference does not change, elements can be added to the 3D scene and projected into the image in an identical way. `model3D_Ro_final` contains the points of a scene with Pikachu and a dinosaur in it.
 
{{< img src="images/pika_dino_blender.png" align="center" title="Adding elements to the 3D scene" >}}
{{< vs 1 >}}

The last lines of the display part are uncommented to get the final result:
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
{{< img src="images/pika_dino.png" align="center" title="New projected scene" >}}
{{< vs 1 >}}

Created by our teacher [Claire Labit-Bonis](https://clairelabitbonis.github.io/).