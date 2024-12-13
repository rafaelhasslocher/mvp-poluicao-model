# MVP-Poluicao-Model

Este repositório contém o MVP desenvolvido para aprovação na Sprint "Machine Learning & Analytics" da Pós-Graduação Lato Sensu em Ciência de Dados e Analytics do CCE PUC-Rio.

Para a construção desse MVP, utilizou-se as bibliotecas `keras` e `stats-models` para ajustar os modelos de interesse. Adicionalmente, as bibliotecas `numpy`, `pandas`, `matplotlib`, `seaborn` são utilizadas para o trabalho da base e visualização dos dados como um todo.

A biblioteca `keras-tuner` é utilizada para a otimização dos hiperparâmetros do modelo LSTM.

A biblioteca `scikit-learn` é utilizada para obter algumas métricas de avaliação dos modelos.

## Utilização

Para treinar o modelo, basta seguir o notebook `poluicao-model.ipynb`.

O notebook contempla a importação dos dados hospedados em um repositório no Github. É importante mencionar que os dados de treino e de teste estão em arquivos diferentes.

Após o tratamento e visualização dos dados, há uma etapa de cross validation de modelos ARIMA para a definição dos hiperparâmetros por meio de métricas como o Akaike Information Criterion (AIC), o Bayesian Information Criterion (BIC) e o Root Mean Squared Error (RMSE).

Em seguida, há um novo tratamento dos dados para adequá-los ao ajuste de uma rede neural recorrente do tipo Long Short Term Memory (LSTM). Nessa etapa, o `keras-tuner` é utilizado para auxiliar na definição dos melhores hiperparâmetros para o ajuste do modelo. 

Por fim, os resultados são comparados com comentários gerais. As funções utilizadas ao longo do código são listadas em sua integridade no início do notebook.