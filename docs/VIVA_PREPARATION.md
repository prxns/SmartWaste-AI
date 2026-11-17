# SmartWaste AI viva preparation

## A two to three minute project explanation

My project is SmartWaste AI, a CNN-based waste classifier combined with Mamdani fuzzy decision support. The aim is to identify the material in a waste image and explain an illustrative recycling suitability assessment.

A user uploads a JPEG or PNG photograph. The application validates its format and size, converts it to RGB, resizes it to the model input, and applies MobileNetV2 preprocessing. An ImageNet-pretrained MobileNetV2 extracts features. A classification head trained on TrashNet predicts cardboard, glass, metal, paper, plastic or trash. The interface shows the top three softmax scores. These scores are not guaranteed probabilities of correctness.

The second stage is separate from image recognition. Each predicted category has a configurable material potential value. These values are academic assumptions, not measured recycling rates. The user manually assesses contamination using a slider. The CNN does not estimate contamination.

The fuzzy engine maps both inputs into Low, Medium and High membership degrees. Nine IF-THEN rules cover all combinations. It takes the minimum of the two antecedent degrees, clips the consequent membership function at that firing strength, and takes the maximum across rules at every output point. Centroid defuzzification gives a crisp suitability score. The laboratory page visualizes input memberships, all firing strengths, the aggregated output and the centroid, so the decision can be inspected.

The training pipeline uses stratified training, validation and test partitions with a fixed seed. Augmentation and class weights apply only to training. Validation loss chooses the best checkpoint; the untouched test split supplies accuracy, macro precision, macro recall, macro F1 and a confusion matrix. I should quote only the results currently saved in the evaluation page, and state clearly if training has not yet run.

The limitations include the small dataset, simple backgrounds, subjective contamination and assumed material values. One mathematical finding is that the exact prescribed membership shapes and centroid operators do not guarantee global monotonicity; small score reversals are documented with counterexamples. This is an academic decision-support prototype, and actual disposal must follow local collection rules.

## Questions and answers

### 1 What is a CNN
A convolutional neural network learns image features using filters shared across spatial locations. Early layers respond to simple patterns; deeper layers combine them into useful representations for classification.

### 2 What is MobileNetV2
MobileNetV2 is an efficient CNN using inverted residual blocks, depthwise convolutions and linear bottlenecks. We reuse an ImageNet-pretrained version as a feature extractor.

### 3 Why choose MobileNetV2
It is relatively small and practical for local inference and transfer learning on a small dataset. It avoids the data and compute cost of training a large CNN from scratch.

### 4 What is transfer learning
Transfer learning reuses features learned on one dataset for a different task. We retain the pretrained convolutional base and train a new six-class head on TrashNet.

### 5 What layers did you add
Global average pooling converts feature maps into a vector, dropout reduces dependence on individual features during training, and a dense softmax layer returns six material scores.

### 6 What is image preprocessing
The image is safely decoded, oriented using EXIF, converted to RGB, resized and passed through `mobilenet_v2.preprocess_input`. This maps raw [0,255] pixels to approximately [-1,1].

### 7 Why must preprocessing match at inference
The model learns from a particular input distribution. A different normalization or channel order during inference can make even valid images produce poor predictions.

### 8 What is data augmentation
Augmentation creates random transformations of training images, such as horizontal flips, small rotations and zooms. Validation and test images are not randomly augmented.

### 9 What is overfitting
Overfitting occurs when the model learns training-specific patterns that do not generalize. Freezing the base, augmentation, dropout, early stopping and validation monitoring help control it.

### 10 What are training validation and test sets
Training adjusts model parameters. Validation selects the best checkpoint and controls learning-rate changes. The test set is isolated until final evaluation.

### 11 Why use a stratified split
It preserves approximately the same class proportions across partitions. That matters because the six TrashNet classes have different numbers of examples.

### 12 Why use class weights
An error on an underrepresented class gets a larger training loss weight. We compute balanced weights from training labels only, rather than duplicating validation or test images.

### 13 How do you prevent data leakage
Exact decoded duplicates are removed before splitting. A seeded split and hash manifest record assignments. Test images do not select epochs or hyperparameters. Near-duplicate leakage is a remaining limitation.

### 14 What is softmax
Softmax turns class logits into nonnegative scores summing to one. The largest score selects the class, but an uncalibrated score is not a guaranteed probability that the prediction is correct.

### 15 What is cross entropy
Cross entropy penalizes low predicted scores for the true class. Sparse categorical cross entropy uses integer class indices, which match our label encoding.

### 16 What is Adam
Adam is a gradient-based optimizer with adaptive parameter update scales. The head starts with a learning rate of 0.001; validation loss may trigger reductions.

### 17 What is early stopping
Training stops after validation loss fails to improve for a configured number of epochs. The best validation checkpoint is used for test evaluation.

### 18 What is a confusion matrix
Rows represent actual classes and columns represent predicted classes. Diagonal counts are correct predictions; other cells show which materials are confused.

### 19 What are precision recall and F1
Precision is true positives divided by all predictions of a class. Recall is true positives divided by actual examples of that class. F1 is their harmonic mean.

### 20 Why report macro F1
Macro F1 averages class F1 scores equally. It prevents a large class from dominating the summary and helps reveal poor performance on the small trash class.

### 21 What is fuzzy logic
Fuzzy logic represents gradual membership in linguistic sets instead of only crisp true/false categories. A contamination value can partly belong to both Low and Medium.

### 22 What is a membership function
A membership function maps a numerical value to a degree between 0 and 1. We use a Low trapezoid, Medium triangle and High trapezoid on [0,100].

### 23 What is fuzzification
Fuzzification calculates input membership degrees. For input 37.5, the specified sets give Low 0.5, Medium 0.5 and High 0.

### 24 What is a fuzzy rule
A fuzzy rule links linguistic input conditions to an output set. For example: IF material potential is High AND contamination is Low THEN suitability is High.

### 25 What is a Mamdani inference system
It evaluates rules with fuzzy consequent sets, combines their implied output shapes and defuzzifies the aggregate. We implement these stages numerically with NumPy.

### 26 What is the difference between Mamdani and Sugeno
Mamdani consequents are fuzzy sets that require output aggregation and defuzzification. Sugeno consequents are typically constants or functions combined by weighted averaging.

### 27 How is AND implemented
The firing strength is the minimum of the material and contamination membership degrees. If the degrees are 0.7 and 0.4, the rule fires at 0.4.

### 28 How are implication and aggregation implemented
Each output membership curve is clipped at the rule firing strength. At each output value, the largest clipped membership across all rules forms the aggregate.

### 29 What is centroid defuzzification
It finds the centre of area under the aggregated membership curve. Numerically we divide its first moment by its area, integrating the sampled piecewise-linear curve.

### 30 Why combine a CNN and fuzzy logic
The CNN handles visual recognition. Fuzzy logic expresses the separate decision-support rules in an inspectable way and combines category assumptions with manual contamination.

### 31 Where exactly does fuzzy logic operate
It runs after CNN classification, using the category's configured potential and the user's contamination input. It does not change CNN weights or generate class probabilities.

### 32 Why is contamination entered manually
TrashNet provides material categories, not contamination labels. Training a contamination estimator would require a separately labelled dataset and validation; claiming it exists would be inaccurate.

### 33 How are material potential values assigned
The project specification supplies illustrative values: cardboard 80, glass 85, metal 90, paper 75, plastic 60 and trash 15. They are editable in `config.py` and are not empirical rates.

### 34 What does the final score represent
It is the centroid of the fuzzy suitability aggregate under our chosen assumptions and inputs. It is not a measured recycling percentage, certification or local acceptance guarantee.

### 35 Does more contamination always lower the score
Not with these exact specified shapes and operators. For potential 15, contamination 25 gives about 19.44 and 26 gives about 19.63. Clipping an asymmetric shoulder moves its centroid. We document this limitation and retain explicit expected-failing monotonicity tests.

### 36 What happens if no rule fires
The aggregate has zero area, so the engine raises a clear error instead of dividing by zero. The supplied overlapping sets cover the normal input domain, and coverage is tested.

### 37 What are the main limitations
The dataset is small and imbalanced, its backgrounds are simple, predictions are uncalibrated, unrelated images are not automatically rejected, contamination is subjective, material assumptions are illustrative, and exact fuzzy monotonicity is not guaranteed.

### 38 How would you improve the project
Collect diverse local images and labelled contamination data, examine near duplicates, calibrate model scores, test rejection of unrelated images, validate expert material rules, and redesign the FIS only with explicit full-grid monotonicity testing.

### 39 What is saved from a classification session
Only an in-memory activity record with category, model score, timestamp, fuzzy inputs and suitability. The photograph is not permanently saved by application code. The user can export the history as CSV or clear it.

### 40 What results should you quote in the viva
Use the current Model Evaluation page and `results/metrics.json`. Quote the test sample count and macro metrics alongside accuracy. If those artifacts are absent, say training or evaluation has not completed; never substitute source-paper or invented numbers.
