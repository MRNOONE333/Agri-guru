import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class PinPage extends StatefulWidget {
  const PinPage({Key? key}) : super(key: key);

  @override
  _PinPageState createState() => _PinPageState();
}

class _PinPageState extends State<PinPage> {
  final TextEditingController _pinController = TextEditingController();
  bool _isLoading = false;

  @override
  void dispose() {
    _pinController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PIN Page'),
        backgroundColor:Color(0xff1C9C73),
      ),
      body: Container(
        color: const Color(0xFFECEEE6),
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextFormField(
              controller: _pinController,
              decoration: const InputDecoration(
                labelText: 'PIN',
              ),
            ),
            const SizedBox(height: 20.0),
            ElevatedButton(
              style:ElevatedButton.styleFrom(
                primary: Color(0xff1C9C73),
              ),
              onPressed: _isLoading ? null : _retrievePassword,
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : const Text('Retrieve Password'),
            ),
          ],
        ),
      ),
    );
  }

  void _retrievePassword() async {
    String pin = _pinController.text;

    setState(() {
      _isLoading = true;
    });

    try {
      // TODO: Implement your Firebase authentication logic here.
      // Replace the code below with the actual Firebase authentication code.
      // For example, you can use Firebase Auth's signInWithEmailAndPassword method.
      UserCredential userCredential = await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: 'user@example.com',
        password: pin,
      );

      // Retrieve the password from the user credential.
      // String password = userCredential.user?.password ?? '';

      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('Password Retrieval'),
            content: Text('The password associated with the provided PIN is: password'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text('OK'),
              ),
            ],
          );
        },
      );
    } catch (e) {
      print('Error retrieving password: $e');
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('Error'),
            content: const Text('Failed to retrieve password. Please try again.'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text('OK'),
              ),
            ],
          );
        },
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }
}

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PIN Page',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: PinPage(),
    );
  }
}