import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SuperMartAnalysis {

    // Data pelanggan dengan atribut usia, jenis kelamin, pendapatan, dan kategori pembelian
    private static Map<String, double[]> customerData = new HashMap<>();

    // Hasil kelompok dari k-means clustering
    private static Map<String, Integer> customerClusters = new HashMap<>();

    // Hasil analisis pembelian dari kolaborasi Naive Bayes
    private static Map<Integer, List<String>> clusterPurchases = new HashMap<>();

    public static void main(String[] args) throws Exception {
        // Mengisi data pelanggan
        populateCustomerData();

        // Melakukan k-means clustering
        kMeansClustering();

        // Melakukan analisis pembelian dengan kolaborasi Naive Bayes
        analyzePurchases();

        // Membangun pohon keputusan berdasarkan hasil analisis
        buildDecisionTree();

        // Melakukan prediksi pengiriman paket berdasarkan pohon keputusan
        predictDelivery("Pelanggan 1", "Elektronik");
        predictDelivery("Pelanggan 2", "Makanan");
        predictDelivery("Pelanggan 3", "Perawatan Kesehatan");
    }

    private static void populateCustomerData() {
        customerData.put("Pelanggan 1", new double[]{25, 1, 2000});
        customerData.put("Pelanggan 2", new double[]{40, 0, 4000});
        customerData.put("Pelanggan 3", new double[]{60, 1, 8000});
        // Tambahkan data pelanggan lainnya jika diperlukan
    }

    private static void kMeansClustering() {
        // Melakukan k-means clustering menggunakan data pelanggan

        // Menentukan jumlah kelompok yang diinginkan
        int k = 3;

        // Menyiapkan data untuk algoritma k-means clustering
        Instances data = prepareDataForClustering();

        // Membangun model k-means clustering
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(k);

        try {
            // Melakukan k-means clustering
            kMeans.buildClusterer(data);

            // Mendapatkan hasil kelompok untuk setiap pelanggan
            for (int i = 0; i < data.numInstances(); i++) {
                int cluster = kMeans.clusterInstance(data.instance(i));
                customerClusters.put(data.instance(i).stringValue(0), cluster);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Instances prepareDataForClustering() {
        // Membuat struktur data untuk k-means clustering
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("NamaPelanggan", (ArrayList<String>) null));
        attributes.add(new Attribute("Usia"));
        attributes.add(new Attribute("JenisKelamin"));
        attributes.add(new Attribute("Pendapatan"));

        Instances data = new Instances("CustomerData", attributes, customerData.size());

        // Mengisi data pelanggan ke dalam struktur data
        for (Map.Entry<String, double[]> entry : customerData.entrySet()) {
            Instance instance = new Instance(attributes.size());
            instance.setValue(attributes.get(0), entry.getKey());
            for (int i = 1; i < attributes.size(); i++) {
                instance.setValue(attributes.get(i), entry.getValue()[i - 1]);
            }
            data.add(instance);
        }

        return data;
    }

    private static void analyzePurchases() {
        // Melakukan analisis pembelian dengan kolaborasi Naive Bayes

        // Menyiapkan data untuk analisis pembelian
        Instances data = prepareDataForNaiveBayes();

        // Membangun model kolaborasi Naive Bayes
        NaiveBayesMultinomial naiveBayes = new NaiveBayesMultinomial();
        try {
            naiveBayes.buildClassifier(data);

            // Mendapatkan hasil analisis pembelian untuk setiap kelompok
            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                int cluster = customerClusters.get(instance.stringValue(0));
                String purchaseCategory = instance.stringValue(instance.numAttributes() - 1);

                if (!clusterPurchases.containsKey(cluster)) {
                    clusterPurchases.put(cluster, new ArrayList<>());
                }
                clusterPurchases.get(cluster).add(purchaseCategory);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Instances prepareDataForNaiveBayes() {
        // Membuat struktur data untuk analisis pembelian
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("NamaPelanggan", (ArrayList<String>) null));
        attributes.add(new Attribute("Usia"));
        attributes.add(new Attribute("JenisKelamin"));
        attributes.add(new Attribute("Pendapatan"));
        attributes.add(new Attribute("KategoriPembelian", (ArrayList<String>) null));

        Instances data = new Instances("CustomerData", attributes, customerData.size());

        // Mengisi data pelanggan dan kategori pembelian ke dalam struktur data
        for (Map.Entry<String, double[]> entry : customerData.entrySet()) {
            int cluster = customerClusters.get(entry.getKey());
            for (String purchaseCategory : clusterPurchases.get(cluster)) {
                Instance instance = new Instance(attributes.size());
                instance.setValue(attributes.get(0), entry.getKey());
                for (int i = 1; i < attributes.size() - 1; i++) {
                    instance.setValue(attributes.get(i), entry.getValue()[i - 1]);
                }
                instance.setValue(attributes.get(attributes.size() - 1), purchaseCategory);
                data.add(instance);
            }
        }

        return data;
    }

    private static void buildDecisionTree() throws Exception {
        // Membangun pohon keputusan berdasarkan hasil analisis

        // Menyiapkan data untuk pembangunan pohon keputusan
        Instances data = prepareDataForDecisionTree();

        // Membangun pohon keputusan
        J48 decisionTree = new J48();
        decisionTree.buildClassifier(data);

        // Menampilkan pohon keputusan
        System.out.println(decisionTree);
    }

    private static Instances prepareDataForDecisionTree() {
        // Membuat struktur data untuk pembangunan pohon keputusan
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("NamaPelanggan", (ArrayList<String>) null));
        attributes.add(new Attribute("Usia"));
        attributes.add(new Attribute("JenisKelamin"));
        attributes.add(new Attribute("Pendapatan"));
        attributes.add(new Attribute("KategoriPembelian", (ArrayList<String>) null));

        Instances data = new Instances("CustomerData", attributes, customerData.size());

        // Mengisi data pelanggan dan kategori pembelian ke dalam struktur data
        for (Map.Entry<String, double[]> entry : customerData.entrySet()) {
            int cluster = customerClusters.get(entry.getKey());
            for (String purchaseCategory : clusterPurchases.get(cluster)) {
                Instance instance = new Instance(attributes.size());
                instance.setValue(attributes.get(0), entry.getKey());
                for (int i = 1; i < attributes.size() - 1; i++) {
                    instance.setValue(attributes.get(i), entry.getValue()[i - 1]);
                }
                instance.setValue(attributes.get(attributes.size() - 1), purchaseCategory);
                data.add(instance);
            }
        }

        return data;
    }

    private static void predictDelivery(String customer, String purchaseCategory) throws Exception {
        // Melakukan prediksi pengiriman paket berdasarkan pohon keputusan

        // Menyiapkan data untuk prediksi
        Instances data = prepareDataForPrediction(customer, purchaseCategory);

        // Membaca pohon keputusan yang telah dibangun
        J48 decisionTree = (J48) weka.core.SerializationHelper.read("decision_tree.model");

        // Melakukan prediksi pengiriman paket
        double prediction = decisionTree.classifyInstance(data.firstInstance());

        // Menampilkan hasil prediksi
        System.out.println("Prediksi pengiriman paket untuk " + customer + " dengan pembelian kategori " + purchaseCategory + ": " + data.classAttribute().value((int) prediction));
    }

    private static Instances prepareDataForPrediction(String customer, String purchaseCategory) {
        // Membuat struktur data untuk prediksi
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("NamaPelanggan", (ArrayList<String>) null));
        attributes.add(new Attribute("Usia"));
        attributes.add(new Attribute("JenisKelamin"));
        attributes.add(new Attribute("Pendapatan"));
        attributes.add(new Attribute("KategoriPembelian", (ArrayList<String>) null));

        Instances data = new Instances("PredictionData", attributes, 1);

        // Mengisi data pelanggan dan kategori pembelian ke dalam struktur data
        Instance instance = new Instance(attributes.size());
        instance.setValue(attributes.get(0), customer);
        instance.setValue(attributes.get(1), customerData.get(customer)[0]);
        instance.setValue(attributes.get(2), customerData.get(customer)[1]);
        instance.setValue(attributes.get(3), customerData.get(customer)[2]);
        instance.setValue(attributes.get(4), purchaseCategory);
        data.add(instance);

        return data;
    }
}
