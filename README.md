# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/68098a9a-e384-4d53-aa89-cc70c8da2a3f)


## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.
### STEP 2:
Model Design:
 * Input Layer: Number of neurons = features.
 * Hidden Layers: 2 layers with ReLU activation.
 * Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.
### STEP 3:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.
### STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.
### STEP 5:
Evaluation: Assess using accuracy, confusion matrix, precision, and recall.
### STEP 6:
Optimization: Tune hyperparameters (layers, neurons, learning rate, batch size).

## PROGRAM

### Name: SARGURU K
### Register Number: 212222230134

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):  # Corrected: __init__ instead of _init
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
        
```
```python
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/a3490a66-6a9c-4f3c-999d-8798d7f34688)


## OUTPUT



### Confusion Matrix

![Screenshot 2025-03-17 083030](https://github.com/user-attachments/assets/bddc91ff-fc38-4244-a0cf-e9bb703ad678)


### Classification Report

![Screenshot 2025-03-17 083150](https://github.com/user-attachments/assets/721c2968-841e-4e52-81a0-25c6fef58be3)



### New Sample Data Prediction

![Screenshot 2025-03-17 083204](https://github.com/user-attachments/assets/95de3dcf-89e8-4e4a-9286-2bab9994a832)


## RESULT
Thus the neural network classification model for the given dataset is developed successfully.
