# Módulo 5 — Transformações Box–Cox

## Seção 3 — Estimação de $\lambda$ por máxima verossimilhança perfilada

**Estado editorial:** seção independente para estudo e revisão científica. **Pré-requisitos:** Módulo 5, Seções 1–2; transformação Box–Cox, distribuição normal, mínimos quadrados, logaritmos e derivadas. **Ambiente:** Python 3, NumPy, SciPy e Matplotlib. Os dados são sintéticos.

**Rastreabilidade.** O `PLANO_MODULOS_PRIORITARIOS.md`, item Módulo 5, solicita a estimação de $\lambda$ por verossimilhança. O `mapeamento-conteudos-series-temporais.md` assinala uma lacuna matemática nas videoaulas sobre Box–Cox. A dedução abaixo é **complementação didática original** a partir do princípio de mudança de variáveis e do modelo normal de regressão. O livro de Morettin e Toloi é referência geral do projeto, mas **não atribuímos esta derivação a páginas desse livro que não foram consultadas para Box–Cox**. A formulação clássica de Box e Cox (1964) é referência bibliográfica adicional, não consultada diretamente nesta elaboração.

### Objetivos

Compreender por que a escolha de $\lambda$ depende do modelo probabilístico; obter o jacobiano da transformação; deduzir a log-verossimilhança perfilada com regressão; calcular um exemplo simples; estimar $\lambda$ sem usar uma função pronta e comparar com SciPy; interpretar intervalos, pressupostos e vazamento temporal.

### 1. Da escolha visual à estimação estatística

A Seção 2 definiu, para $x>0$,

$$
g_\lambda(x)=\begin{cases}(x^\lambda-1)/\lambda,&\lambda\ne0,\\\log x,&\lambda=0.\end{cases}
$$

Uma transformação pode reduzir assimetria ou relação média–variância. Entretanto, **não existe um $\lambda$ universalmente ótimo para os mesmos dados**: o resultado depende de qual variável transformada modelamos, dos regressores e das hipóteses sobre os erros. Em particular, escolher $\lambda$ ignorando a tendência ou sazonalidade pode levar a uma conclusão diferente daquela obtida no modelo que efetivamente será empregado.

### 2. Reforço: o que significa verossimilhança?

Dada uma amostra $\mathbf x=(x_1,\ldots,x_n)$, a função de verossimilhança avalia, como função dos parâmetros, a densidade conjunta atribuída **aos valores observados** pelo modelo. Na maximização, os valores $x_i$ são mantidos fixos. Quando as observações não são independentes, a densidade conjunta não pode ser substituída automaticamente pelo produto de densidades marginais.

Começaremos com erros independentes gaussianos para tornar a dedução explícita. Em séries temporais reais, autocorrelação residual pode invalidar esse modelo simplificado: a solução exige especificar a dependência na verossimilhança, e não ignorá-la.

### 3. Modelo de regressão na escala transformada

Defina $w_i(\lambda)=g_\lambda(x_i)$ e uma matriz de regressão $\mathbf X$ de dimensão $n\times p$, com posto coluna completo. Suponha, **condicionalmente aos regressores considerados fixos**,

$$
\mathbf w_\lambda=\mathbf X\boldsymbol\beta+\boldsymbol\varepsilon,
\qquad \boldsymbol\varepsilon\sim N_n(\mathbf0,\sigma^2\mathbf I_n),\quad\sigma^2>0.
$$

Os símbolos são: $\boldsymbol\beta$ (coeficientes), $\sigma^2$ (variância dos erros transformados), $\mathbf I_n$ (identidade) e $p$ (número de colunas de $\mathbf X$). A hipótese gaussiana conjunta com covariância $\sigma^2\mathbf I_n$ implica independência dos erros nesta especificação.

A densidade **dos valores transformados** é

$$
f_{\mathbf W}(\mathbf w\mid\boldsymbol\beta,\sigma^2,\lambda)
=(2\pi\sigma^2)^{-n/2}
\exp\!\left[-\frac{\sum_{i=1}^n(w_i-\mathbf x_i^\top\boldsymbol\beta)^2}{2\sigma^2}\right].
$$

Mas a amostra observada está na escala **original** $x_i$. Para construir o critério clássico de perfil Box–Cox, precisamos incluir a correção de mudança de variáveis. **Ressalva de rigor:** para $\lambda\ne0$, a imagem de $g_\lambda$ é uma semirreta, ao passo que a distribuição normal irrestrita atribui probabilidade positiva fora dela. Assim, a expressão normal vezes jacobiano apresentada adiante é o **critério clássico de verossimilhança Box–Cox**, mas não é, sem ajuste adicional, uma densidade normalizada em $x>0$. Uma modelagem probabilística exata com normal truncada deve incluir o fator de normalização dependente dos parâmetros, modificando o perfil. Esta seção ensina explicitamente o critério convencional empregado na prática, e não reivindica normalização exata sob o modelo irrestrito.

### 4. Dedução do jacobiano — passo decisivo

A derivada de Box–Cox, demonstrada na Seção 2, é

$$
\frac{d}{dx}g_\lambda(x)=x^{\lambda-1}>0.
$$

Como a transformação é aplicada separadamente em cada observação, a matriz jacobiana de $\mathbf x\mapsto\mathbf w_\lambda$ é diagonal. Seu determinante absoluto é

$$
J_\lambda(\mathbf x)=\prod_{i=1}^n x_i^{\lambda-1}.
$$

Tomando o logaritmo:

$$
\boxed{\log J_\lambda(\mathbf x)=(\lambda-1)\sum_{i=1}^n\log x_i.}
$$

Observe que esse termo **não é opcional**. Sem ele, minimizar simplesmente o erro quadrático de $g_\lambda(x)$ em diferentes escalas favoreceria transformações que reduzem numericamente a amplitude dos dados, em vez de comparar densidades para a mesma amostra observada.

### 5. Log-verossimilhança completa e perfilada

Aplicando a mudança de variáveis ao núcleo gaussiano e incluindo o jacobiano, obtemos a **log-verossimilhança convencional de Box–Cox (sem a constante de truncamento)**:

$$
\ell(\lambda,\boldsymbol\beta,\sigma^2)
=-\frac n2\log(2\pi)-\frac n2\log\sigma^2
-\frac{1}{2\sigma^2}\underbrace{\sum_{i=1}^n\left[g_\lambda(x_i)-\mathbf x_i^\top\boldsymbol\beta\right]^2}_{\mathrm{SSE}(\lambda,\boldsymbol\beta)}
+(\lambda-1)\sum_{i=1}^n\log x_i.
$$

**Passo 1 — maximizar em $\boldsymbol\beta$ para $\lambda$ fixo.** O jacobiano não depende de $\boldsymbol\beta$. Assim, maximizar $\ell$ equivale a minimizar a soma de quadrados:

$$
\widehat{\boldsymbol\beta}(\lambda)=\underset{\boldsymbol\beta}{\arg\min}\,\|\mathbf w_\lambda-\mathbf X\boldsymbol\beta\|_2^2.
$$

Se $\mathbf X$ tem posto completo, a solução matemática é $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf w_\lambda$. Computacionalmente, preferimos `np.linalg.lstsq` sem construir a inversa. Denote $\mathrm{SSE}(\lambda)=\|\mathbf w_\lambda-\mathbf X\widehat{\boldsymbol\beta}(\lambda)\|_2^2$.

**Passo 2 — maximizar em $\sigma^2$.** A derivada da log-verossimilhança com relação a $\sigma^2$ é

$$
\frac{\partial\ell}{\partial\sigma^2}=-\frac{n}{2\sigma^2}+\frac{\mathrm{SSE}(\lambda)}{2(\sigma^2)^2}.
$$

Igualando a zero, para $\mathrm{SSE}(\lambda)>0$:

$$
\boxed{\widehat\sigma^2_{\mathrm{MV}}(\lambda)=\frac{\mathrm{SSE}(\lambda)}{n}.}
$$

Este é o divisor **$n$ da máxima verossimilhança**, não o estimador residual corrigido por $n-p$. São problemas estatísticos diferentes.

**Passo 3 — substituir as estimativas.** Descartando constantes que não dependem de $\lambda$:

$$
\boxed{\ell_p(\lambda)=-\frac n2\log\!\left(\frac{\mathrm{SSE}(\lambda)}n\right)+(\lambda-1)\sum_{i=1}^n\log x_i.}
$$

A estimativa é $\widehat\lambda=\arg\max_{\lambda\in\Lambda}\ell_p(\lambda)$, onde $\Lambda$ é um intervalo de pesquisa definido **antes** da otimização. A hipótese $\mathrm{SSE}(\lambda)>0$ evita uma verossimilhança não limitada quando ocorre ajuste exato. Essa maximização é *perfilada* porque eliminamos analiticamente $\boldsymbol\beta$ e $\sigma^2$, restando uma função de um único parâmetro.

### 6. Exemplo numérico: transformação, resíduos e jacobiano

Considere os dados positivos artificiais $\mathbf x=(2,3,4,7,9)$ e um modelo com **apenas intercepto**. Para $\lambda=0$, os valores transformados são os logaritmos; para $\lambda=1$, são $x_i-1$. Para cada candidato:

1. Calcule $w_i(\lambda)$ e sua média $\overline w_\lambda$.
2. Obtenha $\mathrm{SSE}(\lambda)=\sum_i(w_i-\overline w_\lambda)^2$.
3. Some os logaritmos originais $S=\sum_i\log x_i$.
4. Calcule $-\frac52\log(\mathrm{SSE}/5)+(\lambda-1)S$.

Este último valor é comparável **entre candidatos** porque inclui o jacobiano. O experimento Python abaixo exibe os resultados e permite conferir cada parcela. A amostra de cinco números é um exemplo de cálculo, **não fornece precisão suficiente para selecionar cientificamente uma transformação**.

### 7. Implementação manual primeiro; SciPy depois

O exemplo a seguir usa duas situações: uma amostra pequena para reproduzir as contas e uma amostra sintética com regressores de tendência. A função `perfil_boxcox` implementa literalmente a equação perfilada; `minimize_scalar` maximiza por minimização do negativo da função.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import boxcox_normmax


def transformar_boxcox(x, lam):
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("Box–Cox exige dados positivos e finitos.")
    if lam == 0.0:
        return np.log(x)
    return np.expm1(lam * np.log(x)) / lam


def perfil_boxcox(x, matriz, lam):
    x = np.asarray(x, dtype=float)
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2 or matriz.shape[0] != len(x):
        raise ValueError("A matriz deve ter uma linha por observação.")
    if np.linalg.matrix_rank(matriz) != matriz.shape[1]:
        raise ValueError("A matriz de regressão deve ter posto completo.")
    w = transformar_boxcox(x, lam)
    beta, *_ = np.linalg.lstsq(matriz, w, rcond=None)
    residuos = w - matriz @ beta
    sse = np.dot(residuos, residuos)
    if sse <= 0:
        raise ValueError("A soma de quadrados precisa ser positiva.")
    return -0.5 * len(x) * np.log(sse / len(x)) + (lam - 1) * np.log(x).sum()


amostra = np.array([2., 3., 4., 7., 9.])
X_const = np.ones((len(amostra), 1))
for lam in (0.0, 0.5, 1.0):
    print(f"lambda={lam:4.1f}  log-perfil={perfil_boxcox(amostra, X_const, lam):.5f}")

intervalo = (-2.0, 2.0)
resultado = minimize_scalar(lambda lam: -perfil_boxcox(amostra, X_const, lam),
                            bounds=intervalo, method="bounded")
assert resultado.success
estimativa_manual = resultado.x
estimativa_scipy = boxcox_normmax(amostra, method="mle", brack=(-1, 1))
print("lambda manual (apenas intercepto):", round(estimativa_manual, 5))
print("lambda SciPy (apenas intercepto):", round(estimativa_scipy, 5))
assert abs(estimativa_manual - estimativa_scipy) < 1e-3
```

**Leitura do código.** `np.linalg.lstsq` estima os coeficientes para **cada candidato** $\lambda$; `np.dot` calcula a soma dos quadrados; `np.log(x).sum()` é o logaritmo do jacobiano dividido pelo fator $\lambda-1$; `minimize_scalar(..., method='bounded')` procura o mínimo no intervalo declarado. O resultado de `boxcox_normmax(..., method='mle')` foi comparado **somente no modelo de intercepto**, que corresponde à formulação dessa rotina. Sua utilização direta não ajusta simultaneamente uma regressão temporal arbitrária.

### 8. Por que um modelo de tendência pode mudar $\widehat\lambda$?

Considere $X$ com intercepto e tendência linear. Para cada valor de $\lambda$, o ajuste retira uma tendência da série **já transformada**. A soma de quadrados agora mede dispersão residual ao redor da tendência estimada, e não ao redor de uma constante. A inferência é condicional à adequação dessa tendência e à independência/normalidade assumidas para seus resíduos.

```python
import matplotlib.pyplot as plt

rng = np.random.default_rng(20261118)
n = 180
t = np.arange(n, dtype=float)
# Dados sintéticos lognormais com tendência suave na média logarítmica.
x = np.exp(1.1 + 0.006*t + rng.normal(0, 0.15, n))
X_nivel = np.ones((n, 1))
X_tendencia = np.column_stack([np.ones(n), (t - t.mean()) / t.std()])
lambdas = np.linspace(-1.0, 1.5, 220)
perfil_nivel = np.array([perfil_boxcox(x, X_nivel, l) for l in lambdas])
perfil_tendencia = np.array([perfil_boxcox(x, X_tendencia, l) for l in lambdas])

fit_nivel = minimize_scalar(lambda l: -perfil_boxcox(x, X_nivel, l),
                           bounds=(-1, 1.5), method="bounded")
fit_tend = minimize_scalar(lambda l: -perfil_boxcox(x, X_tendencia, l),
                          bounds=(-1, 1.5), method="bounded")
print("lambda (nível):", round(fit_nivel.x, 4))
print("lambda (com tendência):", round(fit_tend.x, 4))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(lambdas, perfil_nivel - perfil_nivel.max(), label="Somente intercepto")
ax.plot(lambdas, perfil_tendencia - perfil_tendencia.max(), label="Intercepto e tendência")
ax.axvline(fit_nivel.x, linestyle="--", alpha=0.6)
ax.axvline(fit_tend.x, linestyle=":", alpha=0.8)
ax.set(xlabel="Parâmetro lambda", ylabel="Log-perfil menos seu máximo",
       title="Box–Cox: o modelo de média altera o perfil")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
plt.show()
```

![Comparação de perfis Box–Cox com e sem tendência](modulo_05_figuras/04-perfil-verossimilhanca-boxcox.png)

*Figura 3.1 — Perfis de log-verossimilhança normalizados pelo próprio máximo, obtidos sobre dados sintéticos positivos. Modelar a tendência altera o componente residual e pode alterar o parâmetro selecionado. Não se deve interpretar a diferença como um teste formal entre modelos.*

### 9. Incerteza da estimativa e limitações inferenciais

Sob regularidade e uma verossimilhança corretamente especificada, uma região aproximada de perfil de 95% pode ser obtida pelos valores que satisfazem

$$
2\bigl[\ell_p(\widehat\lambda)-\ell_p(\lambda)\bigr]\le\chi^2_{1;0{,}95}\approx3{,}841.
$$

A fórmula usa o resultado assintótico de razão de verossimilhanças e **não garante cobertura exata em amostras pequenas**. Ela requer hipóteses apropriadas, especialmente sobre distribuição e estrutura de dependência dos erros. A escolha do intervalo de busca também deve ser informada: um máximo na borda indica necessidade de investigação adicional.

**Problemas frequentes:** (a) zeros/negativos impossibilitam Box–Cox padrão; (b) deslocar dados muda o problema; (c) tendências, sazonalidade e quebras mal especificadas alteram a estimativa; (d) resíduos serialmente correlacionados invalidam a verossimilhança gaussiana independente usada aqui; (e) máximos rasos indicam incerteza sobre $\lambda$; (f) escolher o parâmetro olhando todo o conjunto de treino e teste provoca **vazamento temporal**.

Em previsão temporal, ajuste $\widehat\lambda$ **dentro de cada janela de treino**. Reaplique a transformação, sem reestimá-la, aos dados de validação quando necessário e respeitando o suporte positivo. `boxcox_normmax` não conhece, por si só, a cronologia da sua série.

### 10. Exercícios orientados

1. **Demonstração:** deduza a expressão do jacobiano por multiplicação das derivadas individuais. Por que não basta usar a SSE para escolher $\lambda$?
2. **Derivação:** derive $\widehat\sigma^2=\mathrm{SSE}/n$ a partir da log-verossimilhança, indicando o papel da hipótese $\mathrm{SSE}>0$.
3. **Numérico:** para $\mathbf x=(2,3,4,7,9)$, calcule $\ell_p(0)$ e $\ell_p(1)$ manualmente, sem otimização.
4. **Pseudocódigo:** descreva como percorrer uma grade de valores de $\lambda$, reestimar $\boldsymbol\beta$ e selecionar o maior perfil antes de usar `minimize_scalar`.
5. **Python:** adicione indicadores trimestrais à matriz de regressão, verifique posto e compare o perfil com o obtido sob apenas intercepto.
6. **Análise científica:** explique por que testar o parâmetro com dados futuros antes da origem de previsão distorce a avaliação de desempenho.

**Pistas:** no exercício 3, lembre-se de que $g_1(x)=x-1$ e $g_0(x)=\log x$; não dispense o termo $(\lambda-1)\sum\log x$. No exercício 5, confirme se todos os grupos sazonais têm observações e evite incluir simultaneamente intercepto e todas as indicadoras.

### 11. Síntese e referências

Escolher $\lambda$ pelo **critério convencional de máxima verossimilhança Box–Cox** significa comparar as densidades gaussianas transformadas **com a correção de jacobiano na escala original**, respeitada a ressalva de suporte e normalização. O jacobiano converte corretamente as densidades e a perfilação elimina os parâmetros de regressão e variância para cada candidato. A seleção depende da especificação dos erros e da média temporal; não é uma propriedade isolada dos números positivos observados.

- **Fonte de planejamento consultada:** `PLANO_MODULOS_PRIORITARIOS.md`, Módulo 5; `mapeamento-conteudos-series-temporais.md`, lacuna Box–Cox.
- **Referência científica complementar, não consultada diretamente:** BOX, G. E. P.; COX, D. R. An analysis of transformations. *Journal of the Royal Statistical Society, Series B*, 26(2), 211–252, 1964.
- **Complementação original:** derivação explícita da verossimilhança, perfil normal com regressão, experimentos, pseudocódigo e discussão de validação cronológica desta seção.
