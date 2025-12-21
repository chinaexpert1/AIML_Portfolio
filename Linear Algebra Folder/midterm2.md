

## __Invertible Matrix Theorem IMT__
- $A$ is an invertible matrix.
- $A^T$ is also invertible.
- There is an $n \times n$ matrix $B$ such that $AB = I = BA$
- $A$ is row-equivalent to the $n × n$ __identity matrix__.
  - $A$ has 'n' pivot positions on the main diagonal == equivalent to  "I"
  - The columns of A form a __linearly independent__ set. ('n' pivot columns)
  - The columns of A span $R^n$ (form basis)
  - null space of A is {0} == rank A is 'n'
- $T(x)=Ax$, $T$ is __one-to-one__.
  - A has 'n' pivot positions every column == square matrix == on the main diagonal == equivalent to  "I"
  - $Ax = 0$ has only trivial solution {# solutions <= 1}
  - The columns of A form a linearly independent set. ('n' pivot columns)
  - __det A $\ne$ 0__, since the columns are independent = the rows are independent $(detA=detA^T)$
- $T(x)=Ax$, T maps __$R^n$ onto $R^n$__
  - A has 'n' pivot positions every row == square matrix == on the main diagonal == equivalent to  "I"
  - The equation Ax=b has {# solutions >= 1} for each b in R^n
  - The columns of A span $R^n$ (form basis)
  - __det A $\ne$ 0__, since the columns are independent = the rows are independent $(detA=detA^T)$
- det A $\ne$ 0
- 0 is not an eigenvalue of A.
 

## __Invertible Matrix (non-singular) Character__
- $A^{-1}$ is unique
- $(A^{-1})^{-1} = A$
- $(A^T)^{-1} = (A^{-1})^T$
- A and B are matrices with AB = I
- $AB = I ~,~ BA = I$
- $(AB)^{-1} = B^{-1}A^{-1}$
- $c\ne0$, $cA$ is invertible then 
  $(cA)^{-1} = \dfrac{A^{-1}}{c}$ 

## __Cramer's Rule__ (invertible matrix)
- $x_i=\dfrac{det~A_i(b)}{det~A}$ 
  - $ex)\\
    3x_1-2x_2=6\\
    -5x_1+4x_2=8\\
    A=\begin{bmatrix}3&-2\\-5&4\end{bmatrix},
    A_1(b)=\begin{bmatrix}6&-2\\8&4\end{bmatrix},
    A_2(b)=\begin{bmatrix}3&6\\-5&8\end{bmatrix}\\
    x_1=\dfrac{det~A_1(b)}{det~A}=\dfrac{40}{2}=20\\
    x_2=\dfrac{det~A_1(b)}{det~A}=\dfrac{54}{2}=27
    $
- $A^{-1}=\dfrac{1}{detA}~adjA~(adj=adjugate=cofactors~of~A)$ 


## __Determinants__
- $det~ A^{-1} = (det~ A)^{-1}$
- $det~ A^T = det~ A$
  - $det(A^T-\lambda I)=det(A^T-\lambda I^T)=det(A-\lambda I)^T=det(A-\lambda I)$
- $det(AB) = (det~ A) (det~ B)$
- $det~A=\begin{bmatrix}a&0&...&0\\0&b&...&0\\0&0&\ddots&0\\0&0&...&n\end{bmatrix}if~diagonal=a*b*...*n=dot~of~eigenvalues/pivots~if~triangular$
- $det A = \dfrac{1}{det~ A^{-1}},~ det~ A^{-1}=\dfrac{1}{det~A}$
 
## __Eigen-stuff__
- $\lambda=eigenvalue~of~A^T=eigenvalue~of~A$ 
- $\lambda^{-1}(=\frac{1}{\lambda})=eigenvalue~of~A^{-1}$ iff $A$ is invertible =$(\lambda\ne0)$.
  - Let $\lambda$ be an eigenvalue of an __invertible__ matrix A. Show $\lambda^{-1}=eigenvalue~of~A^{-1}$
    - $A^{-1}Ax=A^{-1}(\lambda x),~x=A^{-1}(\lambda x),~x=\lambda(A^{-1} x),~\lambda^{-1}x=A^{-1}x$

- The __eigenspace of $A$__ corresponding to $\lambda$ is the __null space__ of the matrix $A-\lambda I$
  - = an eigenspace of $A$ is a null space of a __certain__ matrix
  
## __Diagonalization Theorem__
- $A_{n \times n}$, square matrix, is diagonalizable iff A has __n linearly independent eigenvectors__
  - how to check if there are enough eigenvectors?
  1. all n eigenvalues are __distinct__ 
  2. __if eigenvals are not distinct (n=4, # of eigenval=2)__, still diagonalizable iff
       1. characteristic equation fully factors into linear combo
       2. corresponding dim $(eigenspace_{\lambda_n})$ = multiplicity of $\lambda_n$ 


## __Diagonal $PDP^{-1}$__
- __"Find B-matrix for T"__ = find $[T]_B=D=P^{-1}AP$
- $T:P_2(polynomial) \Rightarrow dim~3(x^0,x^1,x^2),P_3\Rightarrow dim4(x^0,x^1,x^2,x^3)$ 
- "Find a basis $B$ for $R^n$ with the property that $[T]_B$ is diagonal $=~find~P~from~PDP^{-1}~from~A=find~eigenvectors~of~A=P~diagonalizes~A$
- If $A$ is diagonalizable, then so is $A^{-1}$. 
  - $A^{-1}=(PDP^{-1})^{-1}=(P^{-1})^{-1}D^{-1}P^{-1}(note~reversed~order)=PD^{-1}P^{-1}$
- $A$ has __n linearly independent eigenvectors__, then so does $A^T$.
  - $A^T=(PDP^{-1})^T=(P^{-1})^TD^TP^T=(P^T)^{-1}DP^T=QDQ^{-1}, where~Q=(P^T)^{-1}$
- Factorization of $A,~A=PDP^{-1}$ is __not unique__.




# T/F 
### 2.? Rank
- If $A$ is a square matrix, then $Rank(A)=Rank(A^2)$
  - False. consider the matrix $\begin{bmatrix}0&1\\0&0\end{bmatrix}$. $Rank(A)=1,~Rank(A^2)=0~(\begin{bmatrix}0&0\\0&0\end{bmatrix})$

### 3.3 Cramer
- area of parallelogram (2D) = move it to origin (0,0), get 2 basis vectors, then calculate determinant.

### 5.1 eigen-stuff
- if $Ax=\lambda x$ for some scalar $\lambda$, then x is an eigenvector of $A$.
  - false?! will be true iff __x is nonzero vector__... of course..
- A steady-state vector for a stochastic matrix is actually an eigenvector.
  - True. 


### 5.2 characteristic eq
- If $A$ is 3x3, with colmns $a_1,a_2,a_3$, then det$A$ equals the volume of the parallelepiped determined by $a_1,a_2,a_3$.
  - False... -_- need iff $a_1,a_2,a_3$ are __linearly independent__ = invertible.
- A row operation on A does not change the eigenvalues.
  - False. Row operations on a matrix usually change its eigenvalues.

### 5.3 Diagonalization
- $A$ is diagonalizable if $A$ has n eigenvectors
  - False, n __indeoendent__ eigenvectors. -_-
- Every nonzero linear transformation from $R^n$ to itself has at least one nonzero eigenvalue.
  - False, consider the matrix $\begin{bmatrix}0&1\\0&0\end{bmatrix}$. This is nonzero, but its only eigenvalue is zero.
- There exists a real $3 \times 3$ matrix $A$ whose eigenvalues are all (nonreal) complex numbers.
  - False. Nonreal complex eigenvalues come in conjugate pairs $a \pm bi$. if $A$ has only nonreal complex eigenvalues, then the number of rows and columns of $A$ must be __even__.
- There does not exist a 3x3 matrix $A$ with eigenvalues $\lambda=1,-1,-1+i$.
  - True. assume $A$ has real entries, eigenvalues always come in complex conjugate pairs.
- If $A$ is similar to $B$, then $det(A)=det(B)$
  - True if $A=PBP^{-1}$

