import java.util.*;

public class NaiveBayesClassifier {
    private Map<String, Integer> classCounts;
    private Map<String, Map<String, Integer>> wordCounts;
    private Map<String, Double> classProbabilities;

    public NaiveBayesClassifier() {
        classCounts = new HashMap<>();
        wordCounts = new HashMap<>();
        classProbabilities = new HashMap<>();
    }

    public void train(List<String> documents, List<String> classes) {
        // Menghitung jumlah dokumen untuk setiap kelas
        for (String className : classes) {
            classCounts.put(className, classCounts.getOrDefault(className, 0) + 1);
        }

        // Menghitung jumlah kata dalam dokumen untuk setiap kelas
        for (int i = 0; i < documents.size(); i++) {
            String document = documents.get(i);
            String className = classes.get(i);
            Map<String, Integer> counts = wordCounts.getOrDefault(className, new HashMap<>());

            // Memisahkan kata-kata dalam dokumen
            String[] words = document.split("\\s+");

            // Menghitung frekuensi kemunculan kata dalam dokumen
            for (String word : words) {
                counts.put(word, counts.getOrDefault(word, 0) + 1);
            }

            wordCounts.put(className, counts);
        }

        // Menghitung probabilitas kelas
        int totalDocuments = documents.size();
        for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
            String className = entry.getKey();
            int count = entry.getValue();
            double probability = (double) count / totalDocuments;
            classProbabilities.put(className, probability);
        }
    }

    public String classify(String document) {
        // Memisahkan kata-kata dalam dokumen
        String[] words = document.split("\\s+");

        double maxProbability = -1;
        String bestClass = null;

        for (Map.Entry<String, Double> entry : classProbabilities.entrySet()) {
            String className = entry.getKey();
            double classProbability = entry.getValue();
            Map<String, Integer> counts = wordCounts.get(className);

            double logProbability = Math.log(classProbability);

            // Menghitung log probabilitas kelas berdasarkan kata dalam dokumen
            for (String word : words) {
                int wordCount = counts.getOrDefault(word, 0);
                int totalWords = counts.values().stream().mapToInt(Integer::intValue).sum();
                logProbability += Math.log((double) (wordCount + 1) / (totalWords + counts.size()));
            }

            // Memilih kelas dengan probabilitas tertinggi
            if (bestClass == null || logProbability > maxProbability) {
                maxProbability = logProbability;
                bestClass = className;
            }
        }

        return bestClass;
    }

    public static void main(String[] args) {
        // Contoh penggunaan
        List<String> documents = Arrays.asList(
            "Saya suka makan bakso",
            "Saya suka minum teh",
            "Dia adalah seorang dokter",
            "Dia adalah seorang guru"
        );
        List<String> classes = Arrays.asList(
            "Makanan",
            "Makanan",
            "Profesi",
            "Profesi"
        );

        NaiveBayesClassifier classifier = new NaiveBayesClassifier();
        classifier.train(documents, classes);

        String document = "Dia suka minum kopi";
        String predictedClass = classifier.classify(document);
        System.out.println("Dokumen: " + document);
        System.out.println("Kelas prediksi: " + predictedClass);
    }
}
