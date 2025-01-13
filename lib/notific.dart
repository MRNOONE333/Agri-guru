import 'package:flutter/material.dart';

class NotificationsPage extends StatelessWidget {
  const NotificationsPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor:Color(0xff1C9C73),
        title: const Text('Notifications'),
      ),
      body: Container(
        color: const Color(0xFFECEEE6),
        child: Center(
          child: const Text(
            'No notifications',
            style: TextStyle(fontSize: 18.0),
          ),
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
      title: 'Notifications Page',
      theme: ThemeData(
        primarySwatch: Colors.lightGreen,
      ),
      home: NotificationsPage(),
    );
  }
}