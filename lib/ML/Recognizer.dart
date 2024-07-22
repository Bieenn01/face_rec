import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../DB/DatabaseHelper.dart';
import 'Recognition.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  static const int WIDTH = 112;
  static const int HEIGHT = 112;
  final dbHelper = DatabaseHelper();
   @override
  String get modelName => 'mobile_face_net.tflite';

  // Firebase Firestore collection reference
  late CollectionReference<Map<String, dynamic>> recognitionsCollection;

  // Map to store registered recognitions locally
  Map<String, Recognition> registered = {};

  Recognizer({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();
    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    loadModel();
    initDB();

    // Initialize Firebase Firestore collection reference
    recognitionsCollection = FirebaseFirestore.instance.collection('recognitions');
  }

  // Initialize local database (SQLite)
  initDB() async {
    await dbHelper.init();
    loadRegisteredFaces();
  }

  // Load registered faces from SQLite database
  void loadRegisteredFaces() async {
    registered.clear();
    final allRows = await dbHelper.queryAllRows();
    for (final row in allRows) {
      String name = row[DatabaseHelper.columnName];
      List<double> embd = row[DatabaseHelper.columnEmbedding].split(',').map((e) => double.parse(e)).toList().cast<double>();
      Recognition recognition = Recognition(name, Rect.zero, embd, 0);
      registered.putIfAbsent(name, () => recognition);
    }
  }

  // Register face in both SQLite database and Firebase Firestore
  void registerFace(String name, List<double> embedding) async {
    // Register in SQLite
    Map<String, dynamic> row = {
      DatabaseHelper.columnName: name,
      DatabaseHelper.columnEmbedding: embedding.join(",")
    };
    final id = await dbHelper.insert(row);

    // Register in Firestore
    await recognitionsCollection.doc(name).set({
      'name': name,
      'embeddings': embedding,
    }).then((value) {
      print('Face registered in Firebase Firestore: $name');
      loadRegisteredFaces(); // Reload registered faces after registration
    }).catchError((error) {
      print('Failed to register face in Firebase Firestore: $error');
    });
  }

  // Perform face recognition using TensorFlow Lite model
  Recognition recognize(img.Image image, Rect location) {
    var input = imageToArray(image);

    // Output array for inference results
    List output = List.filled(1 * 192, 0).reshape([1, 192]);

    // Perform inference
    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    final run = DateTime.now().millisecondsSinceEpoch - runs;
    print('Time to run inference: $run ms');

    // Convert dynamic list to double list
    List<double> outputArray = output.first.cast<double>();

    // Find the nearest embedding in the registered faces
    Pair pair = findNearest(outputArray);
    print("Distance = ${pair.distance}");

    return Recognition(pair.name, location, outputArray, pair.distance);
  }

  // Convert image to TensorFlow Lite input format
  List<dynamic> imageToArray(img.Image inputImage) {
    img.Image resizedImage = img.copyResize(inputImage!, width: WIDTH, height: HEIGHT);
    List<double> flattenedList = resizedImage.data!.expand((channel) => [channel.r, channel.g, channel.b]).map((value) => value.toDouble()).toList();
    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    int height = HEIGHT;
    int width = WIDTH;
    Float32List reshapedArray = Float32List(1 * height * width * channels);
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] = (float32Array[c * height * width + h * width + w] - 127.5) / 127.5;
        }
      }
    }
    return reshapedArray.reshape([1, 112, 112, 3]);
  }

  // Find the nearest embedding in the registered faces
  Pair findNearest(List<double> emb) {
    Pair pair = Pair("Unknown", -5);
    for (MapEntry<String, Recognition> item in registered.entries) {
      final String name = item.key;
      List<double> knownEmb = item.value.embeddings;
      double distance = 0;
      for (int i = 0; i < emb.length; i++) {
        double diff = emb[i] - knownEmb[i];
        distance += diff * diff;
      }
      distance = sqrt(distance);
      if (pair.distance == -5 || distance < pair.distance) {
        pair.distance = distance;
        pair.name = name;
      }
    }
    return pair;
  }

  // Load TensorFlow Lite model from asset
  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(modelName, options: _interpreterOptions);
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  // Close TensorFlow Lite interpreter
  void close() {
    interpreter.close();
  }
}

class Pair {
  String name;
  double distance;

  Pair(this.name, this.distance);
}
