from functools import lru_cache
import numpy as np

class GaussianNaiveBayes:
  def __init__(self):
    self.classes = None
    self.mean = None
    self.var = None
    self.priors = None

  def fit(self, features, labels):
    self.classes = np.unique(labels)
    n_classes = len(self.classes)
    
    n_samples, n_features = features.shape
        
    # inisiasi mean, variansi, dan priors
    self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
    self.var = np.zeros((n_classes, n_features), dtype=np.float64)
    self.priors = np.zeros(n_classes, dtype=np.float64)

    for class_idx, class_label in enumerate(self.classes):
      class_samples = features[labels == class_label]
      self.mean [class_idx, :] = class_samples.mean(axis=0)
      self.var [class_idx, :] = class_samples.var(axis=0)
      self.priors [class_idx] = class_samples.shape[0] / float(n_samples)
    
  @lru_cache(maxsize=1024)
  def gaussian_pdf (self, class_idx, feature_value):
    mean = self.mean [class_idx]
    var = self.var [class_idx]
    exp = np.exp(-((feature_value - mean) ** 2) / (2 * var))
      
    return (1 / np.sqrt(2 * np.pi * var)) * exp
    
  def posterior (self, sample, class_idx):
    log_prior = np.log(self.priors [class_idx])
    log_likelihood = np.sum(np.log([self.gaussian_pdf(class_idx, feature_value) for feature_value in sample]))
    
    return log_prior + log_likelihood
    
  def predict(self, samples):
    predictions = []
    
    for sample in samples:
      posteriors = [self.log_posterior(sample, idx) for idx in range(len(self.classes))]
      predicted_class_idx = np.argmax(posteriors)
      predictions.append(self.classes[predicted_class_idx])
  
    return np.array(predictions)
      
  def score(self, test_features, true_labels):
    predicted_labels = self.predict(test_features)
    accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)

    return accuracy