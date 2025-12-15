#naive bayes from scartch
import numpy as np

X = np.array([[1.0,1.1],[1.5,1.6],[2.0,1.9],[1.3,0.8],[3.0,3.2],[3.5,2.9],[2.8,3.0],[3.2,3.1]])
y = np.array([0,0,0,0,1,1,1,1])

# Calculate means and variances
class_labels = np.unique(y)
means = {}
variances = {}
for label in class_labels:
    X_class = X[y == label]
    means[label] = np.mean(X_class, axis=0)
    variances[label] = np.var(X_class, axis=0)
# Calculate priors
priors = {label: np.mean(y == label) for label in class_labels}

#gaussian equation=1/√(2πσ²) * e^(-(x-μ)²/2σ²)
def gaussian_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

# Predict function
def predict(X):
    predictions = []
    for x in X:
        class_probs = {}
        for label in class_labels:
            prior = np.log(priors[label])
            likelihood = np.sum(np.log(gaussian_probability(x, means[label], variances[label])))
            class_probs[label] = prior + likelihood
        predicted_label = max(class_probs, key=class_probs.get)
        predictions.append(predicted_label)
    return np.array(predictions)

# Make predictions
y_pred = predict(X)
print("Predicted labels:", y_pred)
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)


