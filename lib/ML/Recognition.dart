import 'dart:ui';
import 'package:cloud_firestore/cloud_firestore.dart';

class Recognition {
  String name;
  Rect location;
  List<double> embeddings;
  double distance;

  Recognition(this.name, this.location, this.embeddings, this.distance);

  // Convert Recognition object to a Map for Firestore
  Map<String, dynamic> toMap() {
    return {
      'name': name,
      'left': location.left,
      'top': location.top,
      'right': location.right,
      'bottom': location.bottom,
      'embeddings': embeddings,
      'distance': distance,
    };
  }

  // Create Recognition object from Firestore document
  static Recognition fromSnapshot(DocumentSnapshot<Map<String, dynamic>> snapshot) {
    var data = snapshot.data()!;
    return Recognition(
      data['name'],
      Rect.fromLTRB(data['left'], data['top'], data['right'], data['bottom']),
      List<double>.from(data['embeddings']),
      data['distance'],
    );
  }
}