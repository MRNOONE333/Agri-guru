import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter1/login.dart';

void main() {
  runApp(const tempo());
}

class tempo extends StatelessWidget {
  const tempo({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "temporary page",
      home: Scaffold(
        body:
        Center(
          child: ElevatedButton(
            onPressed: () {
              FirebaseAuth.instance.signOut().then((value) {
                print("Successfull Logged Out");
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const Login()),
                );
              });
            }, child: const Text("Log out"),
          ),
        ),
      ),
    );
  }
}