import 'package:flutter/material.dart';

class Refer extends StatelessWidget {
  const Refer({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Appbar'),
      ),
      body: Text('Copy this link',
        style: TextStyle(
          color: Colors.black,
        ),),
    );
  }
}

