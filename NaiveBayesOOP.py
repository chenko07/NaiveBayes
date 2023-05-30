import math


class NaiveBayesClassifier:
    def __init__(self):
        self.classes = []
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def train(self, X, y):
        # Menghitung jumlah data pelatihan dalam setiap kelas
        class_counts = {}
        for label in y:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Menghitung probabilitas kelas
        total_samples = len(y)
        self.classes = list(set(y))
        for label in self.classes:
            self.class_probabilities[label] = class_counts[label] / total_samples

        # Menghitung probabilitas fitur dalam setiap kelas
        self.feature_probabilities = {}
        for label in self.classes:
            self.feature_probabilities[label] = {}

            # Menggabungkan semua sampel dengan label yang sama
            samples = [X[i] for i in range(total_samples) if y[i] == label]

            # Menghitung probabilitas fitur untuk setiap atribut
            for i in range(len(X[0])):
                feature_counts = {}
                for sample in samples:
                    feature_value = sample[i]
                    if feature_value not in feature_counts:
                        feature_counts[feature_value] = 0
                    feature_counts[feature_value] += 1

                # Menghitung probabilitas fitur berdasarkan jumlah kemunculannya
                self.feature_probabilities[label][i] = {}
                for feature_value in feature_counts:
                    self.feature_probabilities[label][i][feature_value] = (
                        feature_counts[feature_value] / class_counts[label]
                    )

    def predict(self, X):
        predictions = []
        for sample in X:
            max_probability = -1
            predicted_class = None

            # Mencari kelas dengan probabilitas terbesar
            for label in self.classes:
                probability = self.class_probabilities[label]

                for i in range(len(sample)):
                    feature_value = sample[i]
                    if feature_value in self.feature_probabilities[label][i]:
                        probability *= self.feature_probabilities[label][i][feature_value]
                    else:
                        # Menggunakan smoothing jika nilai fitur tidak ditemukan dalam pelatihan
                        probability *= 0.01

                if probability > max_probability:
                    max_probability = probability
                    predicted_class = label

            predictions.append(predicted_class)

        return predictions


# Contoh penggunaan NaiveBayesClassifier

# Data pelatihan
X_train = [
    [1, 'Sunny'],
    [1, 'Overcast'],
    [2, 'Sunny'],
    [3, 'Rainy'],
    [3, 'Rainy'],
    [3, 'Overcast'],
    [2, 'Overcast'],
    [1, 'Sunny'],
    [1, 'Rainy'],
    [3, 'Sunny']
]
y_train = ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']

# Data pengujian
X_test = [
    [2, 'Sunny'],
    [1, 'Overcast'],
    [3, 'Rainy']
]

# Membuat objek klasifikasi Naive Bayes
nb_classifier = NaiveBayesClassifier()

# Melatih klasifikasi Naive Bayes
nb_classifier.train(X_train, y_train)

# Memprediksi kelas untuk data pengujian
predictions = nb_classifier.predict(X_test)

# Menampilkan hasil prediksi
for i in range(len(X_test)):
    print(f"Data {X_test[i]} diprediksi sebagai {predictions[i]}")
