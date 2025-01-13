import 'package:flutter/material.dart';

class AccountInfoPage extends StatelessWidget {
  final String name;
  final String email;
  final String dob;

  const AccountInfoPage({
    Key? key,
    required this.name,
    required this.email,
    required this.dob,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor:Color(0xff1C9C73),
        title: Text('Account Information'),
      ),
      body: Padding(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Name: $name',
              style: TextStyle(fontSize: 18.0),
            ),
            SizedBox(height: 10.0),
            Text(
              'Email: $email',
              style: TextStyle(fontSize: 18.0),
            ),
            SizedBox(height: 10.0),
            Text(
              'Date of Birth: $dob',
              style: TextStyle(fontSize: 18.0),
            ),
          ],
        ),
      ),
    );
  }
}

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Account Info',
      theme: ThemeData(
        primarySwatch: Colors.lightGreen,
      ),
      home: AccountInfoPage(
        name: 'Ramdas',
        email: 'ramdas@gmail.com',
        dob: '17/1/1965',
      ),
    );
  }
}