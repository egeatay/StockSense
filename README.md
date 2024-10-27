# Stock Sense
### Predictive AI for the Stock Market

This is a research project to experiment with various machine learning methods to see which models can accurately predict the future values of various securities based on past performance.


## Technical Instructions

### Dependencies

You can choose between GPU and CPU use.

#### GPU Use
```bash
conda install keras-gpu=2.6 numpy=1.22.3 tensorflow-gpu=2.6.0 scipy=1.7.3 \
               scikit-learn=1.3.0 seaborn=0.12.2 plotly=5.19.0 pytorch \
               theano=1.0.5 pandas=2.0.3 cudatoolkit=11.3.1

pip install matplotlib==3.7.5
```

#### CPU Use
```bash
conda install keras numpy tensorflow scipy scikit-learn seaborn plotly \
               pytorch theano pandas 

pip install matplotlib torch torchvision
```

### Execution

 - Create a folder named after the stock ticker you wish to analyze and place it in the same directory as other ticker folders.
 - Add the CSV file to the newly created folder.
 - Modify the `for` loop in the script to include your stock's name.
 - Run the model of your choice to test the predictions

### Results

 - All generated graphs will be saved in the `Images` directory.
 - Predictions and test set predictions will be stored in the `PredictionTestSet` directory. Text file will contain the MAE, MSE, RMSE values.



## Authors

This project is a collaborative work of four students:

- [Anthony Barrera](https://github.com/AnthonyBarrera116/)
- [Ege Atay](https://github.com/egeatay/)
- [Pablo Salar Carrera](https://github.com/psalarc)
- Merhi El Kallab

---
